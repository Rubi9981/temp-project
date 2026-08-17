"""주행 루프와 그 부속 상태.

    Shared       — 주행 스레드 <-> 웹 스레드가 주고받는 프레임/텔레메트리
    Profiler     — 루프 단계별 소요시간 측정
    run_loop     — 읽기 -> Driver.step -> HUD/스트리밍/녹화 -> 페이싱
    print_summary— 종료 후 통계 출력
"""
import collections
import os
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

    def report(self):
        if not self.d:
            return ''
        parts, total = [], 0.0
        for k, v in self.d.items():
            arr = np.array(v)
            total += arr.mean()
            parts.append(f'{k}={arr.mean():.1f}/{arr.max():.1f}')
        return f'  [prof ms 평균/최대] ' + '  '.join(parts) + f'   합계={total:.1f}'


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
                    'halted': False, 'frames': 0, 'objects': '-'}


# ==============================================================================
# 루프
# ==============================================================================
def run_loop(hw, driver, shared, args, prof=None, det=None):
    """det 는 yolo.Detector 또는 None. None 이면 객체 탐지를 아예 돌리지 않는다."""
    t0 = time.time()
    last_log = 0
    prof = prof or Profiler(False)

    # 순간 FPS 는 최근 N프레임의 실제 간격으로 잰다. 누적 평균은 프레임이
    # 쌓일수록 둔해져서 주행 중 저하를 못 잡아낸다.
    stamps = collections.deque(maxlen=31)
    pace = (1.0 / args.pace_fps) if args.pace_fps > 0 else 0.0

    try:
        while shared.running:
            loop_start = time.time()

            t = time.perf_counter()
            frame = hw.read()
            prof.add('read', time.perf_counter() - t)
            if frame is None:
                break
            frame = cv2.resize(frame, (cfg.W, cfg.H))

            t = time.perf_counter()
            ctrl, warped, res, y_start = driver.step(frame)
            prof.add('step', time.perf_counter() - t)

            n = driver.stats['frames']

            # 객체 탐지. BEV 가 아니라 원본 카메라 프레임을 그대로 넣는다.
            # 동기 호출이라 이 프레임에서는 추론 시간만큼 루프가 멈춘다 —
            # 그래서 매 프레임이 아니라 --yolo-every 마다만 돌린다.
            if det is not None and n % args.yolo_every == 0:
                t = time.perf_counter()
                det.infer(frame)
                prof.add('yolo', time.perf_counter() - t)

            stamps.append(time.time())
            fps_inst = ((len(stamps) - 1) / (stamps[-1] - stamps[0])
                        if len(stamps) > 1 and stamps[-1] > stamps[0] else 0.0)
            fps_avg = n / max(time.time() - t0, 1e-6)

            tel = {'mode': driver.mode, 'fps': fps_inst, 'fps_avg': fps_avg,
                   'servo': int(round(driver.servo_cmd)), 'motor': driver.motor_cmd,
                   'status': 'OK' if ctrl.ok else 'NO LANE',
                   'halted': driver.stopped, 'frames': n,
                   'objects': det.summary if det is not None else '-'}

            if args.web or args.window:
                t = time.perf_counter()
                vis = hud.overlay(warped, y_start, res, ctrl, tel)
                prof.add('overlay', time.perf_counter() - t)
                if args.web:
                    t = time.perf_counter()
                    ok, buf = cv2.imencode('.jpg', vis,
                                           [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                    prof.add('encode', time.perf_counter() - t)
                    if ok:
                        with shared.lock:
                            shared.jpeg = buf.tobytes()
                            shared.raw = frame
                            shared.tel = tel
                if args.window:
                    cv2.imshow('drive', vis)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                        print('\n창에서 종료')
                        break
            else:
                with shared.lock:
                    shared.tel = tel

            # 저장 형식을 captures/ 와 똑같이 맞춘다 (채널 스왑 후 640x480).
            # 그래야 review.py / evaluate.py 가 동일하게 처리한다.
            if args.record and n % args.record_every == 0:
                t = time.perf_counter()
                cv2.imwrite(os.path.join(args.record, f'frame_{n:06d}.jpg'), frame)
                prof.add('record', time.perf_counter() - t)

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
        try:
            driver.apply_motor(0)
            driver.apply_servo(cfg.SERVO_CENTER)
        except Exception:
            pass


def print_summary(driver, hw, elapsed):
    s = driver.stats
    print()
    print('=' * 52)
    print(f"  프레임 {s['frames']}장  {s['frames'] / max(elapsed, 1e-6):.1f} fps")
    if s['frames']:
        print(f"  조향 산출 {s['ok']} ({100 * s['ok'] / s['frames']:.0f}%)  "
              f"실패 {s['fail']} ({100 * s['fail'] / s['frames']:.0f}%)")
    print(f"  연속 실패로 정지한 횟수: {s['halt']}")
    print('=' * 52)

    if isinstance(hw, hardware.ReplayHardware):
        servos = [v for kind, v in hw.commands if kind == 'servo']
        motors = [v for kind, v in hw.commands if kind == 'motor']
        if servos:
            arr = np.array(servos, float)
            print(f'  servo 명령 {len(servos)}회: mean={arr.mean():.1f} '
                  f'std={arr.std():.1f} 범위 {arr.min():.0f}~{arr.max():.0f}')
            print(f'  motor 0 명령: {motors.count(0)}회 / {len(motors)}회')
