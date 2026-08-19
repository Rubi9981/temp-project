"""주행 루프와 그 부속 상태.

    Shared       — 주행 스레드 <-> 웹 스레드가 주고받는 프레임/텔레메트리
    Profiler     — 루프 단계별 소요시간 측정
    run_loop     — 읽기 -> Driver.step -> HUD/스트리밍/녹화 -> 페이싱
    print_summary— 종료 후 통계 출력
"""
import collections
import os
import queue
import threading
import time

import cv2
import numpy as np

import config as cfg
import hardware
import hud


# ==============================================================================
# 단계별 소요시간 측정
# ==============================================================================
class Profiler:
    """루프 각 단계의 최근 N프레임 소요시간(ms)을 들고 있는다.

    FPS 가 떨어질 때 "어디서" 떨어지는지는 총합만 봐서는 알 수 없다.
    특히 read 가 33ms 근처면 카메라가 블로킹하며 페이싱하고 있다는 뜻이라,
    우리가 추가로 sleep 하면 프레임 경계를 놓쳐 30 -> 15 로 반토막 난다.
    """

    def __init__(self, enabled, window=60):
        self.enabled = enabled
        self.window = window
        self.d = collections.OrderedDict()

    def add(self, name, seconds):
        if not self.enabled:
            return
        self.d.setdefault(name, collections.deque(maxlen=self.window)).append(
            seconds * 1000.0)

    def report(self, title='루프 ms 평균/최대', total_label='한 바퀴'):
        """단계별 평균/최대를 한 줄로. total_label=None 이면 합계를 생략한다.

        루프용과 백그라운드 워커용을 따로 찍으려고 접두사를 인자로 뺐다.
        워커는 서로, 그리고 루프와 **병렬로** 도니까 시간을 더하는 것이
        의미가 없다 — 그래서 백그라운드 줄에서는 합계를 찍지 않는다.
        """
        if not self.d:
            return ''
        parts, total = [], 0.0
        for k, v in self.d.items():
            arr = np.array(v)
            total += arr.mean()
            parts.append(f'{k}={arr.mean():.1f}/{arr.max():.1f}')
        line = f'  [{title}] ' + '  '.join(parts)
        return line if total_label is None else line + f'   {total_label}={total:.1f}'


# ==============================================================================
# 백그라운드 워커
# ==============================================================================
class Worker:
    """한 칸짜리 큐 + 데몬 스레드. 밀리면 새 작업을 **버린다**.

    주행 루프를 지연시킬 수 있는 일(화면 그리기, JPEG 인코딩, YOLO 추론,
    프레임 저장)을 주행 루프 밖으로 빼는 데 쓴다. OpenCV/NCNN 함수는 연산 중 GIL 을 놓으므로
    실제로 다른 코어에서 겹쳐 돈다.

    **쌓지 않고 버리는 것이 핵심이다.** 큐에 모아두면 브라우저에 과거 영상이
    나오고 지연이 계속 늘어난다. 늦은 프레임은 이미 쓸모가 없다.
    """

    def __init__(self, fn, name, prof=None, stage=None):
        self.fn = fn
        self.name = name
        self.prof = prof
        self.stage = stage or name
        self.dropped = 0
        self.done = 0
        self.q = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def offer(self, item):
        """작업을 넘긴다. 워커가 바쁘면 버리고 False 를 돌려준다."""
        try:
            self.q.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            t = time.perf_counter()
            try:
                self.fn(item)
            except Exception as exc:
                # 워커가 죽어도 주행은 계속돼야 한다
                print(f'[{self.name}] 예외: {exc}')
            else:
                self.done += 1
            if self.prof is not None:
                self.prof.add(self.stage, time.perf_counter() - t)

    def stop(self, timeout=2.0):
        try:                                # 자리를 비워야 종료 신호가 들어간다
            self.q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout)


# ==============================================================================
# 공유 상태 (주행 스레드 <-> 웹 스레드)
# ==============================================================================
class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None            # 최신 HUD 프레임 (JPEG bytes)
        self.raw = None             # 최신 원본 프레임 (캡처 저장용)
        self.running = True
        # objects 는 첫 프레임 전에도 웹이 폴링하므로 기본값이 있어야 한다
        # (없으면 화면에 undefined 가 뜬다)
        self.tel = {'mode': 'STOP', 'fps': 0.0, 'fps_avg': 0.0,
                    'servo': cfg.SERVO_CENTER, 'motor': 0, 'status': 'IDLE',
                    'halted': False, 'frames': 0, 'objects': '-', 'link': '-',
                    'mission': '-'}


# ==============================================================================
# 루프
# ==============================================================================
def run_loop(hw, driver, shared, args, prof=None, det=None, mission=None):
    """주행 루프. det 는 yolo.Detector 또는 None.

    mission 은 mission.MissionManager 또는 None. **주행에 관여하지 않는다** —
    상태를 갱신해 화면에 띄우기만 하고, 조향/구동은 driver 가 그대로 한다.

    주행 루프를 지연시킬 수 있는 작업(화면, YOLO, 녹화)은 Worker 로 내보내고
    이 루프에는 read -> step 만 남긴다. YOLO 결과는 비동기로 갱신되지만
    CrossroadDriver의 객체·교차로 판단에는 사용된다. 그래야 서보 갱신 주기가
    일정해진다 — 예전에는 YOLO가 도는 프레임마다 제어 주기가 5배까지 튀었다.
    """
    t0 = time.time()
    last_log = 0
    prof = prof or Profiler(False)
    prof_bg = Profiler(prof.enabled)     # 워커 시간은 따로 잰다 (합계가 섞이지 않게)

    # 순간 FPS 는 최근 N프레임의 실제 간격으로 잰다. 누적 평균은 프레임이
    # 쌓일수록 둔해져서 주행 중 저하를 못 잡아낸다.
    stamps = collections.deque(maxlen=31)
    pace = (1.0 / args.pace_fps) if args.pace_fps > 0 else 0.0

    # --window 는 cv2.imshow 를 써야 하는데 워커 스레드에서 부르면 플랫폼에
    # 따라 불안정하다. 모니터 붙이고 디버깅할 때만 쓰는 옵션이므로 그때는
    # 예전처럼 루프 안에서 그린다 (성능을 포기한다).
    inline_display = args.window
    workers = []

    def _display(item):
        roi, y_start, res, ctrl, tel = item
        vis = hud.overlay(roi, y_start, res, ctrl, tel)
        ok, buf = cv2.imencode('.jpg', vis,
                               [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        if ok:
            with shared.lock:
                shared.jpeg = buf.tobytes()

    disp_w = None
    if args.web and not inline_display:
        disp_w = Worker(_display, 'display', prof_bg)
        workers.append(disp_w)

    yolo_w = None
    if det is not None:
        yolo_w = Worker(det.infer, 'yolo', prof_bg)
        workers.append(yolo_w)

    # 원격 탐지(yolo_remote.RemoteDetector)일 때만 링크 상태가 있다.
    # 로컬 추론에는 끊길 링크가 없으므로 워치독도 없다.
    link_watch = det is not None and hasattr(det, 'link')

    rec_w = None
    if args.record:
        # 저장 형식을 captures/ 와 똑같이 맞춘다 (채널 스왑 후 640x480).
        # 그래야 review.py / evaluate.py 가 동일하게 처리한다.
        rec_w = Worker(lambda it: cv2.imwrite(it[0], it[1]), 'record', prof_bg)
        workers.append(rec_w)

    try:
        while shared.running:
            loop_start = time.time()

            t = time.perf_counter()
            frame = hw.read()
            prof.add('read', time.perf_counter() - t)
            if frame is None:
                break
            frame = cv2.resize(frame, (cfg.W, cfg.H))

            # 워치독은 step 보다 먼저 본다 — 이번 프레임부터 바로 서게.
            # 탐지 결과가 너무 묵었다는 것은 맥과의 링크가 끊겼다는 뜻이고,
            # 그 상태로 계속 달리면 신호등을 못 보고 교차로에 들어간다.
            if link_watch:
                driver.link_halt = not det.link['ok']

            t = time.perf_counter()
            ctrl, roi, res, y_start = driver.step(frame)
            prof.add('step', time.perf_counter() - t)

            n = driver.stats['frames']

            # 미션 상태 갱신. **아직 주행에 연결되어 있지 않다** — 여기서 나온
            # Intent 를 아무도 소비하지 않으므로, mission 을 켜고 끄는 것으로
            # 주행 동작이 달라지지 않는다. 주행에 물릴 때 이 호출이
            # Driver.step() 안으로 옮겨 간다.
            if mission is not None:
                mission.observe(n, ctrl, det)

            # 객체 탐지. BEV 가 아니라 원본 카메라 프레임을 그대로 넘긴다
            # (채널 변환 금지 — yolo.py docstring). 워커가 아직 이전 프레임을
            # 처리 중이면 이번 것은 버린다. --yolo-every 는 "제안 주기"로,
            # NCNN 이 코어를 독점해 제어 루프를 굶기지 않게 하는 조절 손잡이다.
            if yolo_w is not None and n % args.yolo_every == 0:
                yolo_w.offer(frame)

            stamps.append(time.time())
            fps_inst = ((len(stamps) - 1) / (stamps[-1] - stamps[0])
                        if len(stamps) > 1 and stamps[-1] > stamps[0] else 0.0)
            fps_avg = n / max(time.time() - t0, 1e-6)

            tel = {'mode': driver.mode, 'fps': fps_inst, 'fps_avg': fps_avg,
                   'servo': int(round(driver.servo_cmd)), 'motor': driver.motor_cmd,
                   # CrossroadDriver 는 sub_state 로 더 자세히 알려준다
                   'status': getattr(driver, 'sub_state', None)
                             or ('OK' if ctrl.ok else 'NO LANE'),
                   'sub_state': getattr(driver, 'sub_state', ''),
                   'halted': driver.stopped, 'frames': n,
                   'objects': det.summary if det is not None else '-',
                   'link': getattr(det, 'link_str', '-') if det is not None else '-',
                   'mission': mission.status_str if mission is not None else '-'}

            # 텔레메트리와 원본 프레임은 주 루프가 소유한다 — 참조 대입뿐이라
            # 비용이 없고, 화면 워커가 밀려도 상태표는 살아 있다.
            with shared.lock:
                shared.tel = tel
                shared.raw = frame

            if inline_display:
                vis = hud.overlay(roi, y_start, res, ctrl, tel)
                if args.web:
                    ok, buf = cv2.imencode(
                        '.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                    if ok:
                        with shared.lock:
                            shared.jpeg = buf.tobytes()
                cv2.imshow('drive', vis)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    print('\n창에서 종료')
                    break
            elif disp_w is not None:
                disp_w.offer((roi, y_start, res, ctrl, tel))

            if rec_w is not None and n % args.record_every == 0:
                rec_w.offer((os.path.join(args.record, f'frame_{n:06d}.jpg'), frame))

            if args.log_every and n - last_log >= args.log_every:
                last_log = n
                head = f'  [{n:5d}] {fps_inst:4.1f}fps(평균{fps_avg:4.1f}) {driver.mode:6s} '
                if ctrl.ok:
                    print(head + f'servo={tel["servo"]:3d} '
                                 f'delta={ctrl.delta_deg:+5.1f}deg'
                          + ('  CLAMPED' if ctrl.clamped else ''))
                else:
                    print(head + f'검출 실패 ({ctrl.reason})')
                if prof.enabled:
                    print(prof.report())
                    bg = prof_bg.report('백그라운드 ms 평균/최대', None)
                    if bg:
                        drops = '  '.join(f'{w.name}={w.dropped}' for w in workers)
                        print(bg + (f'   버린 프레임 {drops}' if drops else ''))

            # 페이싱. 카메라 get_image() 가 블로킹이면 이미 카메라가 속도를
            # 정하므로 여기서 더 자면 프레임 경계를 놓쳐 반토막 난다.
            # --pace-fps 0 으로 끌 수 있다.
            if pace and not args.replay:
                t = time.perf_counter()
                time.sleep(max(0.0, pace - (time.time() - loop_start)))
                prof.add('sleep', time.perf_counter() - t)

    except KeyboardInterrupt:
        print('\n사용자 종료')
    except Exception as exc:
        print(f'\n[에러] 주행 루프 예외: {exc}')
    finally:
        shared.running = False
        # 모터를 세우는 것이 최우선 — 워커 정리보다 먼저 한다
        try:
            driver.apply_motor(0)
            driver.apply_servo(cfg.SERVO_CENTER)
        except Exception:
            pass
        for w in workers:
            w.stop()


def print_summary(driver, hw, elapsed, mission=None):
    s = driver.stats
    print()
    print('=' * 52)
    print(f"  프레임 {s['frames']}장  {s['frames'] / max(elapsed, 1e-6):.1f} fps")
    if s['frames']:
        print(f"  조향 산출 {s['ok']} ({100 * s['ok'] / s['frames']:.0f}%)  "
              f"실패 {s['fail']} ({100 * s['fail'] / s['frames']:.0f}%)")
    if 'crossroad' in s:        # CrossroadDriver 일 때만
        print(f"  교차로 직진 {s['crossroad']}  객체 정지 {s['object_stop']}")
    print(f"  연속 실패로 정지한 횟수: {s['halt']}")
    print('=' * 52)

    if mission is not None:
        # 어디서 왜 상태가 바뀌었는지. 주행에 관여하지 않으므로 이 이력이
        # 상태 기계를 확인하는 유일한 창구다.
        print(f'  [미션] 마지막 상태 {mission.status_str}')
        print(mission.format_history())
        print('=' * 52)

    if isinstance(hw, hardware.ReplayHardware):
        servos = [v for kind, v in hw.commands if kind == 'servo']
        motors = [v for kind, v in hw.commands if kind == 'motor']
        if servos:
            arr = np.array(servos, float)
            print(f'  servo 명령 {len(servos)}회: mean={arr.mean():.1f} '
                  f'std={arr.std():.1f} 범위 {arr.min():.0f}~{arr.max():.0f}')
            print(f'  motor 0 명령: {motors.count(0)}회 / {len(motors)}회')
