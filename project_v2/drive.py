"""라즈베리파이 실시간 주행 루프 + 웹 디버그 대시보드.

    afb1.camera.get_image()
      -> 채널 스왑 -> BEV warp -> 이진화 -> 슬라이딩 윈도우 -> 중심선
      -> Pure Pursuit -> afb1.gpio.servo() / motor()

디버그 화면은 **자체 Flask 서버**로 띄운다 (raspi/L_5_Capture.py 방식).
afb1.flask 는 쓰지 않는다 — 동작을 확인할 수 없어 화면이 안 뜰 위험이 있고,
직접 띄우면 주소와 포트를 우리가 안다.

    http://<Pi주소>:5000

afb1 API 는 raspi/ 예제에서 확인된 것만 쓴다:
    gpio.init() / gpio.stby(1) / gpio.servo(30~150) / gpio.motor(speed) / gpio.stop_all()
    camera.init(w, h, fps) / camera.get_image() / camera.release_camera()

사용:
    python3 drive.py --dry-run                      # 모터 끈 채 확인 (첫 브링업)
    python3 drive.py --speed 40 --record run1       # 주행 + 녹화 (웹 자동 활성)
    python3 drive.py --replay ../project/captures   # 하드웨어 없이 로직 검증

안전:
    - 기본은 모터 정지. --speed 를 명시해야 움직인다
    - 웹에서 STOP / 비상정지 버튼으로 즉시 세울 수 있다
    - 연속 MAX_FAIL_FRAMES 프레임 검출 실패 시 자동 정지
    - Ctrl+C / 예외 / 정상 종료 어느 쪽이든 finally 에서 stop_all()
"""
import argparse
import dataclasses
import os
import threading
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

    def __init__(self):
        try:
            import afb1
        except ImportError as exc:
            raise SystemExit(
                'afb1 모듈을 찾을 수 없습니다. 이 스크립트는 라즈베리파이에서 실행해야 합니다.\n'
                '개발 PC에서 루프 로직만 확인하려면: python3 drive.py --replay <이미지폴더>'
            ) from exc
        self.afb1 = afb1

        afb1.gpio.init()
        # raspi/L_5_Capture.py 가 주행 전에 호출한다. 모터 드라이버 standby 해제로
        # 보이며, 빠뜨리면 모터가 돌지 않을 수 있다.
        afb1.gpio.stby(1)
        afb1.camera.init(cfg.W, cfg.H, cfg.CAMERA_FPS)

    def read(self):
        frame = self.afb1.camera.get_image()
        if frame is None:
            return None
        if frame.ndim == 3 and frame.shape[2] == 4:      # RGBA 로 오는 경우 대비
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # raspi/L_5_Capture.py 가 저장 전에 하던 채널 스왑을 그대로 재현한다.
        # captures/ 의 이미지는 이 스왑을 거친 뒤 imwrite 된 것이라, 여기서
        # 똑같이 하지 않으면 오프라인에서 튜닝한 이진화가 다른 색을 보게 된다.
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def servo(self, angle):
        self.afb1.gpio.servo(int(angle))

    def motor(self, speed):
        self.afb1.gpio.motor(int(speed))

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
        self.commands = []          # (kind, value) 기록 — 사후 점검용

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

    def shutdown(self):
        pass


# ==============================================================================
# HUD
# ==============================================================================
def overlay(warped, y_start, res, ctrl, tel):
    """BEV 위에 주행 상태를 그린다.

    cv2.putText 는 Hershey 폰트라 ASCII 만 그릴 수 있다.
    """
    vis = warped.copy()
    h, w = vis.shape[:2]

    # ROI 가이드 + 화면 중앙 기준선
    cv2.rectangle(vis, (0, y_start), (w - 1, h - 1), (255, 120, 0), 2)
    cv2.line(vis, (w // 2, y_start), (w // 2, h), (0, 0, 255), 1)

    # 좌우 차선 다항식
    for fit, color in ((res.fit_left, (255, 128, 0)), (res.fit_right, (0, 128, 255))):
        if fit is not None:
            pts = bevlib.curve_points(fit, 0, h - y_start,
                                      y_offset=y_start).astype(np.int32)
            cv2.polylines(vis, [pts], False, color, 2)

    # 주행 목표 경로
    if res.fit_center is not None:
        pts = bevlib.curve_points(res.fit_center, 0, h - y_start,
                                  y_offset=y_start).astype(np.int32)
        cv2.polylines(vis, [pts], False, (0, 0, 0), 6)
        cv2.polylines(vis, [pts], False, (0, 255, 255), 3)

    # Pure Pursuit 목표점
    if ctrl.ok:
        gx, gy = ctrl.goal_bev
        cv2.circle(vis, (int(gx), int(gy)), 10, (0, 0, 0), -1)
        cv2.circle(vis, (int(gx), int(gy)), 8, (255, 0, 255), -1)

    # 좌상단 상태
    mode_color = {'RUN': (0, 255, 0), 'STOP': (0, 0, 255)}.get(tel['mode'], (0, 200, 255))
    rows = [
        (f"MODE: {tel['mode']}", mode_color),
        (f"FPS: {tel['fps']:.1f}", (200, 200, 200)),
        (f"SERVO: {tel['servo']}", (200, 200, 200)),
        (f"MOTOR: {tel['motor']}", (200, 200, 200)),
    ]
    for i, (text, color) in enumerate(rows):
        y = 28 + i * 26
        cv2.putText(vis, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,
                    cv2.LINE_AA)

    # 우상단 검출/제어 상태
    if ctrl.ok:
        status = f"{res.status.upper()}  d={ctrl.delta_deg:+.1f}deg  Ld={ctrl.lookahead_cm:.0f}cm"
        status_color = (0, 255, 255)
    else:
        status = 'NO LANE'
        status_color = (0, 0, 255)
    if tel['halted']:
        status = 'HALTED - ' + status
        status_color = (0, 0, 255)
    (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    sx = max(w - tw - 14, 200)      # 좌상단 상태 열과 겹치지 않게
    cv2.putText(vis, status, (sx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(vis, status, (sx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1,
                cv2.LINE_AA)

    # 하단 조향 게이지 — 서보 30~150 을 -100~+100 px 로 매핑
    cx, gy = w // 2, h - 22
    cv2.line(vis, (cx - 100, gy), (cx + 100, gy), (90, 90, 90), 3)
    cv2.line(vis, (cx, gy - 7), (cx, gy + 7), (150, 150, 150), 2)
    span = (cfg.SERVO_MAX - cfg.SERVO_MIN) / 2.0
    offset = int((tel['servo'] - cfg.SERVO_CENTER) * (100.0 / span))
    offset = int(np.clip(offset, -100, 100))
    cv2.circle(vis, (cx + offset, gy), 9, (0, 0, 0), -1)
    cv2.circle(vis, (cx + offset, gy), 7, (0, 255, 0), -1)
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

        self.mode = 'RUN'           # 'RUN' | 'STOP' — 웹에서 바꾼다
        self.servo_cmd = float(cfg.SERVO_CENTER)
        self.motor_cmd = 0
        self.fail_streak = 0
        self.stopped = False
        self.stats = {'frames': 0, 'ok': 0, 'fail': 0, 'halt': 0}

    def step(self, frame):
        """한 프레임 처리. (ctrl, warped, res, y_start) 를 돌려준다."""
        self.stats['frames'] += 1

        warped, _ = bevlib.warp_image(frame)
        roi, y_start = bevlib.roi_of(warped)
        mask = self.bin_fn(roi)
        res = self.det_fn(mask)
        ctrl = self.pp(res, roi.shape[0], y_start)

        if self.mode == 'STOP':
            # 인지는 계속 돌린다 — 화면은 살아 있어야 상태를 볼 수 있다
            self.motor_cmd = 0
            self.hw.motor(0)
            return ctrl, warped, res, y_start

        if ctrl.ok:
            self.stats['ok'] += 1
            self.fail_streak = 0
            self.stopped = False
            # 지수이동평균으로 프레임 간 튀는 명령을 완화한다
            target = ctrl.servo
            if self.invert_servo:
                target = 2 * cfg.SERVO_CENTER - target
            self.servo_cmd += self.alpha * (target - self.servo_cmd)
            self.motor_cmd = self.speed
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
                self.motor_cmd = 0
                self.hw.motor(0)
            else:
                # 짧은 끊김은 직전 명령을 유지하고 넘어간다
                self.motor_cmd = self.speed
                self.hw.servo(round(self.servo_cmd))
                self.hw.motor(self.speed)

        return ctrl, warped, res, y_start


# ==============================================================================
# 공유 상태 (주행 스레드 <-> 웹 스레드)
# ==============================================================================
class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None            # 최신 HUD 프레임 (JPEG bytes)
        self.raw = None             # 최신 원본 프레임 (캡처 저장용)
        self.running = True
        self.tel = {'mode': 'RUN', 'fps': 0.0, 'servo': cfg.SERVO_CENTER,
                    'motor': 0, 'status': 'IDLE', 'halted': False, 'frames': 0}


# ==============================================================================
# 웹 대시보드
# ==============================================================================
PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<title>Lane Tracer</title>
<style>
 body{background:#111;color:#ddd;font-family:system-ui,sans-serif;margin:0;padding:16px}
 h1{font-size:18px;margin:0 0 12px}
 .wrap{display:flex;gap:16px;flex-wrap:wrap}
 img{border:1px solid #333;max-width:100%;height:auto}
 .panel{min-width:240px}
 table{border-collapse:collapse;font-size:14px;width:100%}
 td{padding:4px 8px;border-bottom:1px solid #262626}
 td:first-child{color:#888}
 button{font-size:15px;padding:10px 16px;margin:4px 4px 0 0;border:0;
        border-radius:6px;color:#fff;cursor:pointer}
 .run{background:#1a7f37}.stop{background:#8b6b00}.estop{background:#b62324}
 .cap{background:#30363d}
</style></head><body>
<h1>Lane Tracer — live debug</h1>
<div class="wrap">
  <img src="/video_feed" width="640">
  <div class="panel">
    <table>
      <tr><td>MODE</td><td id="mode">-</td></tr>
      <tr><td>FPS</td><td id="fps">-</td></tr>
      <tr><td>SERVO</td><td id="servo">-</td></tr>
      <tr><td>MOTOR</td><td id="motor">-</td></tr>
      <tr><td>STATUS</td><td id="status">-</td></tr>
      <tr><td>FRAMES</td><td id="frames">-</td></tr>
    </table>
    <div>
      <button class="run"   onclick="cmd('run')">START</button>
      <button class="stop"  onclick="cmd('stop')">STOP</button>
    </div>
    <div>
      <button class="estop" onclick="cmd('estop')">EMERGENCY STOP</button>
      <button class="cap"   onclick="cmd('capture')">CAPTURE</button>
    </div>
  </div>
</div>
<script>
function cmd(a){fetch('/api/control',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify({action:a})});}
setInterval(async()=>{
  const r=await fetch('/api/status'); const s=await r.json();
  for(const k of ['mode','servo','motor','status','frames'])
    document.getElementById(k).textContent=s[k];
  document.getElementById('fps').textContent=s.fps.toFixed(1);
},300);
</script></body></html>
"""


def make_app(shared, driver, save_dir):
    try:
        from flask import Flask, Response, jsonify, request
    except ImportError as exc:
        raise SystemExit(
            'Flask 가 없어 웹 디버그 화면을 띄울 수 없습니다.\n'
            '  설치: sudo apt install python3-flask   (또는 pip install flask)\n'
            '  웹 없이 돌리려면: --no-web'
        ) from exc

    app = Flask(__name__)

    @app.route('/')
    def index():
        return PAGE

    def mjpeg():
        while shared.running:
            with shared.lock:
                data = shared.jpeg
            if data is not None:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
            time.sleep(0.033)          # 약 30fps

    @app.route('/video_feed')
    def video_feed():
        return Response(mjpeg(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/api/status')
    def status():
        with shared.lock:
            return jsonify(dict(shared.tel))

    @app.route('/api/control', methods=['POST'])
    def api_control():
        action = (request.get_json(silent=True) or {}).get('action')

        if action == 'run':
            driver.mode = 'RUN'
            driver.fail_streak = 0
            driver.stopped = False
        elif action in ('stop', 'estop'):
            driver.mode = 'STOP'
            try:
                driver.hw.motor(0)
                if action == 'estop':
                    driver.servo_cmd = float(cfg.SERVO_CENTER)
                    driver.hw.servo(cfg.SERVO_CENTER)
            except Exception as exc:
                print(f'[web] 정지 명령 실패: {exc}')
            print(f'  [web] {action.upper()}')
        elif action == 'capture':
            with shared.lock:
                raw = None if shared.raw is None else shared.raw.copy()
            if raw is not None:
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, f'cap_{time.strftime("%Y%m%d_%H%M%S")}.jpg')
                cv2.imwrite(path, raw)
                print(f'  [web] 캡처 저장: {path}')

        return jsonify({'ok': True, 'mode': driver.mode})

    return app


# ==============================================================================
# 루프
# ==============================================================================
def run_loop(hw, driver, shared, args):
    t0 = time.time()
    last_log = 0
    frame_period = 1.0 / cfg.CAMERA_FPS

    try:
        while shared.running:
            loop_start = time.time()

            frame = hw.read()
            if frame is None:
                break
            frame = cv2.resize(frame, (cfg.W, cfg.H))

            ctrl, warped, res, y_start = driver.step(frame)

            n = driver.stats['frames']
            fps_now = n / max(time.time() - t0, 1e-6)

            tel = {'mode': driver.mode, 'fps': fps_now,
                   'servo': int(round(driver.servo_cmd)), 'motor': driver.motor_cmd,
                   'status': 'OK' if ctrl.ok else 'NO LANE',
                   'halted': driver.stopped, 'frames': n}

            if args.web or args.window:
                vis = overlay(warped, y_start, res, ctrl, tel)
                if args.web:
                    ok, buf = cv2.imencode('.jpg', vis,
                                           [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
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
                cv2.imwrite(os.path.join(args.record, f'frame_{n:06d}.jpg'), frame)

            if args.log_every and n - last_log >= args.log_every:
                last_log = n
                if ctrl.ok:
                    print(f'  [{n:5d}] {fps_now:4.1f}fps  {driver.mode:4s} '
                          f'servo={tel["servo"]:3d} delta={ctrl.delta_deg:+5.1f}deg'
                          + ('  CLAMPED' if ctrl.clamped else ''))
                else:
                    print(f'  [{n:5d}] {fps_now:4.1f}fps  {driver.mode:4s} '
                          f'검출 실패 ({ctrl.reason})')

            # 카메라 프레임레이트에 맞춰 쉰다. 웹 스레드가 쓸 CPU 를 남긴다.
            if not args.replay:
                time.sleep(max(0.001, frame_period - (time.time() - loop_start)))

    except KeyboardInterrupt:
        print('\n사용자 종료')
    except Exception as exc:
        print(f'\n[에러] 주행 루프 예외: {exc}')
    finally:
        shared.running = False
        try:
            hw.motor(0)
            hw.servo(cfg.SERVO_CENTER)
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

    if isinstance(hw, ReplayHardware):
        servos = [v for kind, v in hw.commands if kind == 'servo']
        motors = [v for kind, v in hw.commands if kind == 'motor']
        if servos:
            arr = np.array(servos, float)
            print(f'  servo 명령 {len(servos)}회: mean={arr.mean():.1f} '
                  f'std={arr.std():.1f} 범위 {arr.min():.0f}~{arr.max():.0f}')
            print(f'  motor 0 명령: {motors.count(0)}회 / {len(motors)}회')


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
    # 디버그 화면
    ap.add_argument('--no-web', action='store_true', help='웹 대시보드 끄기')
    ap.add_argument('--port', type=int, default=5000)
    ap.add_argument('--jpeg-quality', type=int, default=75, help='스트리밍 JPEG 품질')
    ap.add_argument('--window', action='store_true',
                    help='cv2.imshow 창도 띄우기 (Pi에 모니터/VNC 가 있을 때)')
    # 튜닝
    ap.add_argument('--invert-servo', action='store_true',
                    help='서보가 반대로 돌 때. --dry-run 으로 확인 후 사용')
    ap.add_argument('--steer-gain', type=float, default=cfg.SERVO_PER_DEG,
                    help='서보단위/바퀴각(deg). 조향이 과하면 줄이고 모자라면 키운다')
    ap.add_argument('--center-offset', type=float, default=0.0,
                    help='BEV상 차량 중심선 보정(px). + 면 좌선회를 더 한다 '
                         '= 차가 오른쪽으로 쏠릴 때 쓴다')
    # 데이터
    ap.add_argument('--record', metavar='DIR',
                    help='주행 프레임을 저장. 나중에 review.py 로 되돌려 본다')
    ap.add_argument('--record-every', type=int, default=1, help='N프레임마다 저장')
    ap.add_argument('--replay', metavar='DIR', help='하드웨어 대신 저장 이미지 사용')
    ap.add_argument('--replay-loop', action='store_true', help='replay 반복')
    ap.add_argument('--log-every', type=int, default=30, help='N프레임마다 상태 출력')
    args = ap.parse_args()

    # replay 는 기본적으로 웹을 안 띄운다 (로직 검증용). 필요하면 --port 로 켠다.
    args.web = not args.no_web
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
        hw = Afb1Hardware()

    pp = control.PurePursuit(metric=metric, lookahead_cm=args.lookahead,
                             servo_per_deg=args.steer_gain)
    driver = Driver(hw, pp,
                    binarize.BACKENDS[args.binarize],
                    detect.DETECTORS[args.detect],
                    speed, args.max_fail, args.ema, args.invert_servo)
    shared = Shared()

    if args.record:
        os.makedirs(args.record, exist_ok=True)
        print(f'[record] {args.record}/ 에 저장 (매 {args.record_every}프레임)')

    print(f'binarize={args.binarize} detect={args.detect} Ld={args.lookahead}cm '
          f'speed={speed} ema={args.ema} steer_gain={args.steer_gain:.2f}')
    if speed == 0:
        print('모터 속도 0 — 조향만 계산합니다. 주행하려면 --speed 를 주세요.')

    t0 = time.time()

    if not args.web:
        run_loop(hw, driver, shared, args)
    else:
        app = make_app(shared, driver,
                       save_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'captures'))
        worker = threading.Thread(target=run_loop, args=(hw, driver, shared, args),
                                  daemon=True)
        worker.start()

        print()
        print('=' * 52)
        print(f'  웹 디버그 화면: http://<Pi주소>:{args.port}')
        print('  START / STOP / EMERGENCY STOP / CAPTURE 버튼 제공')
        print('=' * 52)
        print('Ctrl+C 로 종료\n')
        try:
            app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
        except KeyboardInterrupt:
            print('\n사용자 종료')
        finally:
            shared.running = False
            worker.join(timeout=2.0)

    if args.window:
        cv2.destroyAllWindows()
    hw.shutdown()
    print_summary(driver, hw, time.time() - t0)


if __name__ == '__main__':
    main()
