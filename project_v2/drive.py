"""라즈베리파이 실시간 주행 루프.

    afb1.camera.get_image()
      -> 채널 스왑 -> BEV warp -> 이진화 -> 슬라이딩 윈도우 -> 중심선
      -> Pure Pursuit -> afb1.gpio.servo() / motor()

afb1 API 는 raspi/ 의 예제에서 확인된 것만 쓴다:
    gpio.init() / gpio.stby(1) / gpio.servo(30~150) / gpio.motor(speed) / gpio.stop_all()
    camera.init(w, h, fps) / camera.get_image() / camera.release_camera()
    flask.imshow(name, frame, 0)     (선택, 원격 미리보기)

사용:
    python3 drive.py --dry-run          # 하드웨어 붙이고 모터만 끈 채 확인 (권장 첫 단계)
    python3 drive.py --speed 60         # 실제 주행
    python3 drive.py --replay ../project/captures    # 하드웨어 없이 루프 로직만 검증

안전:
    - 기본은 모터 정지 상태다. --speed 를 명시해야 움직인다
    - 연속 MAX_FAIL_FRAMES 프레임 검출 실패 시 모터를 세운다
    - Ctrl+C / 예외 / 정상 종료 어느 쪽이든 finally 에서 stop_all()
"""
import argparse
import dataclasses
import os
import time

import cv2
import numpy as np

import bev as bevlib
import binarize
import config as cfg
import control
import detect


# ==============================================================================
# 하드웨어 추상화
# ==============================================================================
class Afb1Hardware:
    """실제 afb1 하드웨어."""

    def __init__(self, preview=False, preview_name='AFB Camera'):
        try:
            import afb1
        except ImportError as exc:
            raise SystemExit(
                'afb1 모듈을 찾을 수 없습니다. 이 스크립트는 라즈베리파이에서 실행해야 합니다.\n'
                '개발 PC에서 루프 로직만 확인하려면: python3 drive.py --replay <이미지폴더>'
            ) from exc
        self.afb1 = afb1
        self.preview = preview
        # raspi/L_1_camera.py 가 쓰는 이름. afb1.flask 가 이름으로 스트림을
        # 구분한다면 기존에 보던 링크와 같은 이름이어야 화면에 나온다.
        self.preview_name = preview_name

        afb1.gpio.init()
        # raspi/L_5_Capture.py 가 주행 전에 호출한다. 모터 드라이버 standby 해제로
        # 보이며, 빠뜨리면 모터가 돌지 않을 수 있다.
        afb1.gpio.stby(1)
        afb1.camera.init(cfg.W, cfg.H, cfg.CAMERA_FPS)

    def read(self):
        frame = self.afb1.camera.get_image()
        if frame is None:
            return None
        # raspi/L_5_Capture.py 가 저장 전에 하던 채널 스왑을 그대로 재현한다.
        # captures/ 의 이미지는 이 스왑을 거친 뒤 imwrite 된 것이라, 여기서
        # 똑같이 하지 않으면 오프라인에서 튜닝한 이진화가 다른 색을 보게 된다.
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def servo(self, angle):
        self.afb1.gpio.servo(int(angle))

    def motor(self, speed):
        self.afb1.gpio.motor(int(speed))

    def show(self, name, image):
        """afb1.flask 로 프레임을 쏜다.

        추가 채널 변환을 하지 않는다. L_1_camera.py 는
        cvtColor(get_image(), BGR2RGB) 를 flask.imshow 에 넘기는데,
        read() 가 이미 같은 변환을 거친 배열을 돌려주므로 여기서 한 번 더
        뒤집으면 원래대로 돌아가 색이 반대로 나온다.

        다만 오버레이 색은 BGR 튜플로 그렸으므로 브라우저에서는 R/B 가
        바뀌어 보인다 (노란 중심선이 하늘색으로). 배경 영상의 색을 제대로
        보는 쪽이 판단에 중요해서 이렇게 뒀다.
        """
        if not self.preview:
            return
        try:
            self.afb1.flask.imshow(self.preview_name, image, 0)
        except Exception as exc:          # 미리보기 실패가 주행을 멈추면 안 된다
            self.preview = False
            print(f'[preview] 비활성화: {exc}')

    def shutdown(self):
        try:
            self.afb1.gpio.stop_all()
        finally:
            try:
                self.afb1.camera.release_camera()
            except Exception:
                pass


class ReplayHardware:
    """하드웨어 없이 저장 이미지를 먹여 루프 로직만 검증한다.

    실차에서 확인할 수 없는 것(카메라, 서보, 모터)을 뺀 나머지 — 상태 기계,
    실패 카운팅, 평활, 종료 처리 — 를 개발 PC에서 돌려볼 수 있게 한다.
    """

    def __init__(self, directory, loop=False):
        import glob
        self.paths = sorted(
            p for p in glob.glob(os.path.join(directory, '*'))
            if p.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        if not self.paths:
            raise SystemExit(f'이미지가 없습니다: {directory}')
        self.i = 0
        self.loop = loop
        self.commands = []          # (servo, speed) 기록 — 사후 점검용

    def read(self):
        if self.i >= len(self.paths):
            if not self.loop:
                return None
            self.i = 0
        img = cv2.imread(self.paths[self.i])
        self.i += 1
        return img

    def servo(self, angle):
        self.commands.append(('servo', int(angle)))

    def motor(self, speed):
        self.commands.append(('motor', int(speed)))

    def show(self, name, image):
        pass

    def shutdown(self):
        pass


def overlay(warped, y_start, res, ctrl, servo_cmd, fps):
    """미리보기용 최소 오버레이. 브링업 때 눈으로 확인할 것만 그린다.

    review.py 를 임포트하지 않는다 — 주행 루프에 필요 없는 코드를 Pi 로
    끌고 갈 이유가 없고, 매 프레임 그리는 비용도 제어 루프가 쓸 CPU 다.
    """
    vis = warped.copy()
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, y_start), (w - 1, h - 1), (255, 255, 0), 1)
    cv2.line(vis, (w // 2, y_start), (w // 2, h), (0, 0, 255), 1)

    if res.fit_center is not None:
        pts = bevlib.curve_points(res.fit_center, 0, h - y_start,
                                  y_offset=y_start).astype(np.int32)
        cv2.polylines(vis, [pts], False, (0, 255, 255), 3)

    if ctrl.ok:
        gx, gy = ctrl.goal_bev
        cv2.circle(vis, (int(gx), int(gy)), 9, (255, 0, 255), -1)
        txt = f'servo={round(servo_cmd):3d} delta={ctrl.delta_deg:+5.1f} {fps:4.1f}fps'
        color = (255, 255, 255)
    else:
        txt = f'NO LANE  {fps:4.1f}fps'
        color = (0, 0, 255)
    cv2.putText(vis, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(vis, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return vis


# ==============================================================================
# 주행 루프
# ==============================================================================
class Driver:
    def __init__(self, hw, pp, bin_fn, det_fn, speed,
                 max_fail=None, ema_alpha=None, invert_servo=False):
        self.hw = hw
        self.pp = pp
        self.bin_fn = bin_fn
        self.det_fn = det_fn
        self.speed = speed
        self.max_fail = cfg.MAX_FAIL_FRAMES if max_fail is None else max_fail
        self.alpha = cfg.SERVO_EMA_ALPHA if ema_alpha is None else ema_alpha
        self.invert_servo = invert_servo

        self.servo_cmd = float(cfg.SERVO_CENTER)
        self.fail_streak = 0
        self.stopped = False
        self.stats = {'frames': 0, 'ok': 0, 'fail': 0, 'halt': 0}

    def step(self, frame):
        """한 프레임 처리. (ctrl, warped, res, y_start) 를 돌려준다."""
        self.stats['frames'] += 1

        frame = cv2.resize(frame, (cfg.W, cfg.H))
        warped, _ = bevlib.warp_image(frame)
        roi, y_start = bevlib.roi_of(warped)
        mask = self.bin_fn(roi)
        res = self.det_fn(mask)
        ctrl = self.pp(res, roi.shape[0], y_start)

        if ctrl.ok:
            self.stats['ok'] += 1
            self.fail_streak = 0
            self.stopped = False
            # 지수이동평균으로 프레임 간 튀는 명령을 완화한다
            target = ctrl.servo
            if self.invert_servo:
                target = 2 * cfg.SERVO_CENTER - target
            self.servo_cmd += self.alpha * (target - self.servo_cmd)
            self.hw.servo(round(self.servo_cmd))
            self.hw.motor(self.speed)
        else:
            self.stats['fail'] += 1
            self.fail_streak += 1
            if self.fail_streak >= self.max_fail:
                # 차선을 계속 못 찾으면 직전 조향을 유지한 채 세운다.
                # 모르는 상태로 계속 달리는 것보다 안전하다.
                if not self.stopped:
                    self.stats['halt'] += 1
                    print(f'  [정지] 연속 {self.fail_streak}프레임 검출 실패')
                self.stopped = True
                self.hw.motor(0)
            else:
                # 짧은 끊김은 직전 명령을 유지하고 넘어간다
                self.hw.servo(round(self.servo_cmd))
                self.hw.motor(self.speed)

        return ctrl, warped, res, y_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=int, default=0,
                    help=f'모터 속도 (기본 0 = 정지). 주행하려면 명시. 권장 {cfg.DRIVE_SPEED}')
    ap.add_argument('--dry-run', action='store_true',
                    help='서보만 움직이고 모터는 항상 0 — 첫 브링업용')
    ap.add_argument('--binarize', default='adaptive', choices=list(binarize.BACKENDS))
    ap.add_argument('--detect', default='sliding', choices=list(detect.DETECTORS))
    ap.add_argument('--lookahead', type=float, default=cfg.LOOKAHEAD_CM)
    ap.add_argument('--max-fail', type=int, default=cfg.MAX_FAIL_FRAMES)
    ap.add_argument('--ema', type=float, default=cfg.SERVO_EMA_ALPHA,
                    help='서보 평활 계수. 1.0 = 평활 없음')
    ap.add_argument('--preview', action='store_true',
                    help='afb1.flask 로 원격 미리보기 (SSH 접속 상태에서 브라우저로 봄)')
    ap.add_argument('--preview-name', default='AFB Camera',
                    help='afb1.flask.imshow 에 넘길 이름. 기존에 보던 스트림과 '
                         '같은 이름이어야 그 링크에 나온다 (L_1_camera.py 기본값)')
    ap.add_argument('--window', action='store_true',
                    help='cv2.imshow 로 창 띄우기 (Pi에 모니터/VNC 가 있을 때)')
    ap.add_argument('--invert-servo', action='store_true',
                    help='서보가 반대로 돌 때. --dry-run 으로 확인 후 사용')
    ap.add_argument('--steer-gain', type=float, default=cfg.SERVO_PER_DEG,
                    help='서보단위/바퀴각(deg). 조향이 과하면 줄이고 모자라면 키운다')
    ap.add_argument('--center-offset', type=float, default=0.0,
                    help='BEV상 차량 중심선 보정(px). + 면 좌선회를 더 한다 '
                         '= 차가 오른쪽으로 쏠릴 때 쓴다. 근본 해결은 '
                         '직선 차선 정중앙에 세우고 실제 중심 x 를 재는 것')
    ap.add_argument('--record', metavar='DIR',
                    help='주행 프레임을 저장. 나중에 review.py 로 되돌려 본다')
    ap.add_argument('--record-every', type=int, default=1, help='N프레임마다 저장')
    ap.add_argument('--replay', metavar='DIR', help='하드웨어 대신 저장 이미지 사용')
    ap.add_argument('--replay-loop', action='store_true', help='replay 반복')
    ap.add_argument('--log-every', type=int, default=30, help='N프레임마다 상태 출력')
    args = ap.parse_args()

    speed = 0 if args.dry_run else args.speed

    metric = cfg.get_metric()
    if not metric.measured:
        print('[경고] metric.json 이 없습니다. 종방향 스케일이 추정값이라 조향각이')
        print('       실제와 다를 수 있습니다. calibrate_metric.py 를 먼저 돌리세요.')

    if args.center_offset:
        metric = dataclasses.replace(
            metric, vehicle_center_x_px=metric.vehicle_center_x_px + args.center_offset)
        print(f'[보정] vehicle_center_x_px = {metric.vehicle_center_x_px:.1f}')

    if args.replay:
        hw = ReplayHardware(args.replay, loop=args.replay_loop)
        print(f'[replay] {len(hw.paths)}장 — 하드웨어 없이 루프 로직만 검증')
    else:
        hw = Afb1Hardware(preview=args.preview, preview_name=args.preview_name)

    pp = control.PurePursuit(metric=metric, lookahead_cm=args.lookahead,
                             servo_per_deg=args.steer_gain)
    driver = Driver(hw, pp,
                    binarize.BACKENDS[args.binarize],
                    detect.DETECTORS[args.detect],
                    speed, args.max_fail, args.ema, args.invert_servo)

    if args.record:
        os.makedirs(args.record, exist_ok=True)
        print(f'[record] {args.record}/ 에 저장 (매 {args.record_every}프레임)')

    print(f'binarize={args.binarize} detect={args.detect} Ld={args.lookahead}cm '
          f'speed={speed} ema={args.ema} steer_gain={args.steer_gain:.2f}')
    if speed == 0:
        print('모터 속도 0 — 조향만 계산합니다. 주행하려면 --speed 를 주세요.')
    if args.preview:
        print(f"[preview] afb1.flask 로 '{args.preview_name}' 이름으로 송출합니다.")
    if args.window:
        print('창 종료: q 또는 ESC')
    print('Ctrl+C 로 종료\n')

    t0 = time.time()
    last_log = 0
    try:
        while True:
            frame = hw.read()
            if frame is None:
                break

            frame = cv2.resize(frame, (cfg.W, cfg.H))
            ctrl, warped, res, y_start = driver.step(frame)

            n = driver.stats['frames']
            fps_now = n / max(time.time() - t0, 1e-6)

            if args.preview or args.window:
                vis = overlay(warped, y_start, res, ctrl, driver.servo_cmd, fps_now)
                if args.preview:
                    hw.show('lane', vis)
                if args.window:
                    cv2.imshow('drive', vis)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                        print('\n창에서 종료')
                        break

            # 저장 형식을 captures/ 와 똑같이 맞춘다 (채널 스왑 후 640x480).
            # 그래야 review.py / evaluate.py 가 동일하게 처리한다.
            if args.record and n % args.record_every == 0:
                cv2.imwrite(os.path.join(args.record, f'frame_{n:06d}.jpg'), frame)

            if args.log_every and n - last_log >= args.log_every:
                last_log = n
                fps = fps_now
                if ctrl.ok:
                    print(f'  [{n:5d}] {fps:4.1f}fps  servo={round(driver.servo_cmd):3d} '
                          f'delta={ctrl.delta_deg:+5.1f}deg'
                          + ('  CLAMPED' if ctrl.clamped else ''))
                else:
                    print(f'  [{n:5d}] {fps:4.1f}fps  검출 실패 ({ctrl.reason})')

    except KeyboardInterrupt:
        print('\n사용자 종료')
    finally:
        # 어떤 경로로 끝나든 반드시 세운다
        try:
            hw.motor(0)
            hw.servo(cfg.SERVO_CENTER)
        except Exception:
            pass
        if args.window:
            cv2.destroyAllWindows()
        hw.shutdown()

        s = driver.stats
        elapsed = time.time() - t0
        print()
        print('=' * 52)
        print(f"  프레임 {s['frames']}장  {s['frames'] / max(elapsed, 1e-6):.1f} fps")
        if s['frames']:
            print(f"  조향 산출 {s['ok']} ({100 * s['ok'] / s['frames']:.0f}%)  "
                  f"실패 {s['fail']} ({100 * s['fail'] / s['frames']:.0f}%)")
        print(f"  연속 실패로 정지한 횟수: {s['halt']}")
        print('=' * 52)

        if isinstance(hw, ReplayHardware):
            servos = [v for kind, v in hw.commands if kind == 'servo']
            motors = [v for kind, v in hw.commands if kind == 'motor']
            if servos:
                arr = np.array(servos, float)
                print(f'  servo 명령 {len(servos)}회: mean={arr.mean():.1f} '
                      f'std={arr.std():.1f} 범위 {arr.min():.0f}~{arr.max():.0f}')
                print(f'  motor 0 명령: {motors.count(0)}회 / {len(motors)}회')


if __name__ == '__main__':
    main()
