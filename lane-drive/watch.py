"""카메라 전체 화면 + 객체 탐지 + 수동 주행.

세 가지만 한다.

    1. 카메라 원본 화면을 그대로 웹으로 보여준다 (BEV 로 펴지 않는다)
    2. 그 위에 YOLO 탐지 박스를 그린다
    3. 방향키로 직접 몬다

차선 인지(BEV/이진화/슬라이딩 윈도우/Pure Pursuit)는 **전혀 돌리지 않는다.**
자율주행도 없다. 눈으로 보면서 몰고, 무엇이 탐지되는지 확인하는 용도다.

    drive.py    차선 자율주행 (+ 탐지는 숫자로만)
    collect.py  수동 주행 + 프레임 저장
    watch.py    수동 주행 + 탐지 화면          <- 이 파일

하드웨어는 hardware.py, 추론과 박스 그리기는 yolo.py 를 그대로 쓴다.

사용:
    python3 watch.py                                 # 매 프레임 탐지
    python3 watch.py --detect-every 3 --imgsz 448    # 화면을 부드럽게
    python3 watch.py --replay ../images/obstacles --replay-loop

    http://<Pi주소>:5002

조작 (브라우저):
    방향키 또는 화면 버튼 — **누르는 동안만** 이동하고 떼면 중립으로 돌아온다
    DETECT — 탐지 켜기/끄기. 끄면 화면이 빨라진다
    비상정지 — 모터 정지 + 서보 중립

추론과 화면 그리기는 각각 loop.Worker 로 나가 **주 루프와 별도 스레드에서
돈다.** 워커가 밀리면 그 프레임은 버려지고 화면은 직전 박스를 그대로 쓰므로,
탐지를 켜도 영상이 멈추지 않는다. 그래도 느리면 --detect-every 를 늘린다.

안전:
    - 모터는 0, 서보는 중립으로 시작한다. 키를 눌러야만 움직인다
    - 키를 떼거나 브라우저 탭을 벗어나면 즉시 중립으로 돌아간다
    - Ctrl+C / 예외 / 정상 종료 어느 쪽이든 finally 에서 중립 + stop_all()
"""
import argparse
import collections
import threading
import time

import cv2

import config as cfg
import hardware
import loop as looplib          # Profiler 만 빌려 쓴다 (drive.py 와 같은 출력 형식)
import yolo


# ==============================================================================
# 수동 조작
# ==============================================================================
class Manual:
    """방향키 명령을 하드웨어로 옮긴다.

    하드웨어는 루프 스레드와 웹 스레드가 함께 만지므로 hw_lock 으로 직렬화한다
    (driver.Driver / collect.Collector 와 같은 이유, 같은 방식).
    """

    def __init__(self, hw, speed):
        self.hw = hw
        self.speed = speed
        self.hw_lock = threading.Lock()
        self.servo_cmd = cfg.SERVO_CENTER
        self.motor_cmd = 0
        self.detecting = True

    def apply_servo(self, angle):
        self.servo_cmd = int(min(max(angle, cfg.SERVO_MIN), cfg.SERVO_MAX))
        with self.hw_lock:
            self.hw.servo(self.servo_cmd)

    def apply_motor(self, speed):
        self.motor_cmd = int(speed)
        with self.hw_lock:
            self.hw.motor(self.motor_cmd)

    def neutral(self):
        """모터 정지 + 서보 중립. 비상정지와 종료 처리가 함께 쓴다."""
        self.apply_motor(0)
        self.apply_servo(cfg.SERVO_CENTER)

    def on_key(self, key, down):
        """방향키 눌림/뗌. 떼면 반드시 중립으로 — key_up 을 놓치면 계속 달린다."""
        if down:
            if key == 'ArrowUp':
                self.apply_motor(self.speed)
            elif key == 'ArrowDown':
                self.apply_motor(-self.speed)
            elif key == 'ArrowLeft':
                self.apply_servo(cfg.SERVO_MIN)
            elif key == 'ArrowRight':
                self.apply_servo(cfg.SERVO_MAX)
        else:
            if key in ('ArrowUp', 'ArrowDown'):
                self.apply_motor(0)
            elif key in ('ArrowLeft', 'ArrowRight'):
                self.apply_servo(cfg.SERVO_CENTER)


class Shared:
    """루프 스레드 <-> 웹 스레드."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
        self.running = True
        self.tel = {'detect': True, 'objects': '-', 'fps': 0.0, 'frames': 0,
                    'servo': cfg.SERVO_CENTER, 'motor': 0, 'link': '-'}


# ==============================================================================
# 웹 화면
# ==============================================================================
PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<title>Lane Tracer — watch</title>
<style>
 body{background:#111;color:#ddd;font-family:system-ui,sans-serif;margin:0;padding:16px}
 h1{font-size:18px;margin:0 0 12px}
 .wrap{display:flex;gap:16px;flex-wrap:wrap}
 img{border:1px solid #333;max-width:100%;height:auto}
 .panel{min-width:260px}
 table{border-collapse:collapse;font-size:14px;width:100%}
 td{padding:4px 8px;border-bottom:1px solid #262626}
 td:first-child{color:#888}
 #objects{color:#7ee787}
 button{font-size:15px;padding:10px 16px;margin:4px 4px 0 0;border:0;
        border-radius:6px;color:#fff;cursor:pointer}
 .det{background:#1a7f37}.det.off{background:#8b6b00}
 .estop{background:#b62324}
 #pad{margin-top:14px}
 #pad button{width:74px;height:56px;font-size:22px;background:#30363d}
 .hint{color:#888;font-size:13px;margin-top:8px;line-height:1.5}
</style></head><body>
<h1>Lane Tracer — 카메라 화면 + 객체 탐지 + 수동 주행</h1>
<div class="wrap">
  <img src="/video_feed" width="640">
  <div class="panel">
    <table>
      <tr><td>DETECT</td><td id="detect">-</td></tr>
      <tr><td>탐지된 것</td><td id="objects">-</td></tr>
      <tr><td>FPS</td><td id="fps">-</td></tr>
      <tr><td>FRAMES</td><td id="frames">-</td></tr>
      <tr><td>SERVO</td><td id="servo">-</td></tr>
      <tr><td>MOTOR</td><td id="motor">-</td></tr>
      <tr><td>LINK</td><td id="link">-</td></tr>
    </table>
    <div>
      <button class="det" id="b_det" onclick="toggleDet()">DETECT</button>
      <button class="estop" onclick="cmd('estop')">EMERGENCY STOP</button>
    </div>

    <div id="pad">
      <div style="text-align:center">
        <button data-k="ArrowUp">&#9650;</button></div>
      <div style="text-align:center">
        <button data-k="ArrowLeft">&#9664;</button>
        <button data-k="ArrowDown">&#9660;</button>
        <button data-k="ArrowRight">&#9654;</button></div>
      <div class="hint">키보드 방향키로도 조종할 수 있습니다.<br>
        <b>누르는 동안만</b> 움직이고 떼면 멈춥니다.<br>
        영상이 느리면 DETECT 를 꺼 보세요.</div>
    </div>
  </div>
</div>
<script>
function post(body){return fetch('/api/control',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
function cmd(a){return post({action:a});}
function keyDown(k){return post({action:'key_down',key:k});}
function keyUp(k){return post({action:'key_up',key:k});}

let det=true;
function toggleDet(){det=!det; post({action:'detect',on:det});}

// 화면 버튼 — 누르는 동안 유지
for(const b of document.querySelectorAll('#pad button')){
  const k=b.dataset.k;
  const down=e=>{e.preventDefault();keyDown(k);};
  const up=e=>{e.preventDefault();keyUp(k);};
  b.addEventListener('mousedown',down); b.addEventListener('mouseup',up);
  b.addEventListener('mouseleave',up);
  b.addEventListener('touchstart',down,{passive:false});
  b.addEventListener('touchend',up,{passive:false});
}

// 키보드 방향키. 오토리피트로 중복 전송되지 않게 눌린 키를 추적한다.
const held=new Set();
const KEYS=['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'];
addEventListener('keydown',e=>{
  if(!KEYS.includes(e.key)||held.has(e.key))return;
  e.preventDefault(); held.add(e.key); keyDown(e.key);});
addEventListener('keyup',e=>{
  if(!KEYS.includes(e.key))return;
  e.preventDefault(); held.delete(e.key); keyUp(e.key);});
// 탭을 벗어나면 눌린 키를 모두 놓아준다 (키업을 놓치면 계속 달린다)
addEventListener('blur',()=>{for(const k of held)keyUp(k); held.clear();});

setInterval(async()=>{
  const r=await fetch('/api/status'); const s=await r.json();
  for(const k of ['objects','frames','servo','motor','link'])
    document.getElementById(k).textContent=s[k];
  document.getElementById('fps').textContent=s.fps.toFixed(1);
  document.getElementById('detect').textContent=s.detect?'켜짐':'꺼짐';
  det=s.detect;
  document.getElementById('b_det').classList.toggle('off',!s.detect);
},300);
</script></body></html>
"""


def make_app(shared, man):
    try:
        from flask import Flask, Response, jsonify, request
    except ImportError as exc:
        raise SystemExit(
            'Flask 가 없어 화면을 띄울 수 없습니다.\n'
            '  설치: sudo apt install python3-flask   (또는 pip install flask)'
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
        data = request.get_json(silent=True) or {}
        action = data.get('action')

        try:
            if action in ('key_down', 'key_up'):
                man.on_key(data.get('key'), action == 'key_down')

            elif action == 'estop':
                man.neutral()
                print('  [web] EMERGENCY STOP')

            elif action == 'detect':
                man.detecting = bool(data.get('on', True))
                print(f"  [web] 탐지 {'켜짐' if man.detecting else '꺼짐'}")

        except Exception as exc:                # 요청 하나가 서버를 죽이면 안 된다
            print(f'[web] 명령 처리 실패 ({action}): {exc}')
            return jsonify({'ok': False}), 500

        return jsonify({'ok': True, 'detect': man.detecting})

    return app


# ==============================================================================
# 루프
# ==============================================================================
def run_loop(hw, man, det, shared, args, prof=None):
    """카메라를 읽어 워커에 넘기는 것만 한다.

    추론과 화면(박스 그리기 + JPEG 인코딩)은 각각 별도 스레드로 나간다.
    그래야 탐지를 켜도 영상이 끊기지 않아 조종이 가능하다 —
    느린 영상으로 차를 모는 것 자체가 위험하기 때문이다.
    """
    stamps = collections.deque(maxlen=31)
    frames = 0
    last_log = 0
    prof = prof or looplib.Profiler(False)
    prof_bg = looplib.Profiler(prof.enabled)

    def _display(item):
        frame, detecting = item
        vis = yolo.draw_boxes(frame, det.boxes) if detecting else frame
        ok, buf = cv2.imencode('.jpg', vis,
                               [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        if ok:
            with shared.lock:
                shared.jpeg = buf.tobytes()

    disp_w = looplib.Worker(_display, 'display', prof_bg)
    yolo_w = looplib.Worker(det.infer, 'yolo', prof_bg)
    workers = [disp_w, yolo_w]

    try:
        while shared.running:
            t = time.perf_counter()
            frame = hw.read()
            prof.add('read', time.perf_counter() - t)
            if frame is None:
                print('\n더 읽을 프레임이 없습니다')
                break
            frame = cv2.resize(frame, (cfg.W, cfg.H))
            frames += 1

            # 탐지. 프레임을 그대로 넘긴다 (채널 변환 금지 — yolo.py docstring).
            # 워커가 바쁘면 버린다. 화면은 직전 박스를 그대로 쓴다.
            if man.detecting and frames % args.detect_every == 0:
                yolo_w.offer(frame)
            disp_w.offer((frame, man.detecting))

            stamps.append(time.time())
            fps = ((len(stamps) - 1) / (stamps[-1] - stamps[0])
                   if len(stamps) > 1 and stamps[-1] > stamps[0] else 0.0)

            with shared.lock:
                shared.tel = {'detect': man.detecting,
                              'objects': det.summary if man.detecting else '-',
                              'fps': fps, 'frames': frames,
                              'servo': man.servo_cmd, 'motor': man.motor_cmd,
                              'link': getattr(det, 'link_str', '-')}

            if args.log_every and frames - last_log >= args.log_every:
                last_log = frames
                print(f'  [{frames:5d}] {fps:4.1f}fps  '
                      f"{'DET' if man.detecting else '---'}  "
                      f'{det.summary if man.detecting else "-":24s} '
                      f'servo={man.servo_cmd:3d} motor={man.motor_cmd:4d}')
                if prof.enabled:
                    print(prof.report())
                    bg = prof_bg.report('백그라운드 ms 평균/최대', None)
                    if bg:
                        drops = '  '.join(f'{w.name}={w.dropped}' for w in workers)
                        print(bg + f'   버린 프레임 {drops}')

    except KeyboardInterrupt:
        print('\n사용자 종료')
    except Exception as exc:
        print(f'\n[에러] 루프 예외: {exc}')
    finally:
        shared.running = False
        # 모터를 세우는 것이 최우선 — 워커 정리보다 먼저 한다
        try:
            man.neutral()
        except Exception:
            pass
        for w in workers:
            w.stop()


def main():
    ap = argparse.ArgumentParser(
        description='카메라 화면 + 객체 탐지 + 수동 주행')
    ap.add_argument('--detect-every', type=int, default=1, metavar='N',
                    help='N프레임마다 추론 (기본 1 = 매 프레임). 영상이 느리면 늘린다')
    ap.add_argument('--model', default=cfg.YOLO_MODEL_PATH, metavar='PT')
    ap.add_argument('--conf', type=float, default=cfg.YOLO_CONF)
    ap.add_argument('--imgsz', type=int, default=cfg.YOLO_IMGSZ,
                    help='추론 입력 크기. 낮추면 빠르지만 탐지를 잃는다 '
                         '(448=90%%, 320=73%%)')
    # 원격 추론 — Pi4 로컬 YOLO 가 느려서 맥에 맡긴다. 상대편은 yolo_server.py.
    ap.add_argument('--yolo-remote', metavar='HOST[:PORT]',
                    help='맥의 추론 서버로 프레임을 보내 원격 추론한다. '
                         '이때 Pi 에는 ultralytics 도 모델 파일도 필요 없다')
    ap.add_argument('--yolo-jpeg-quality', type=int, default=cfg.YOLO_JPEG_QUALITY,
                    help=f'전송용 JPEG 품질 (기본 {cfg.YOLO_JPEG_QUALITY})')
    ap.add_argument('--yolo-timeout', type=float, default=cfg.YOLO_TIMEOUT_S,
                    help=f'왕복 최대 대기 시간(초, 기본 {cfg.YOLO_TIMEOUT_S})')
    ap.add_argument('--speed', type=int, default=cfg.DRIVE_SPEED,
                    help=f'전진/후진 모터 속도 (기본 {cfg.DRIVE_SPEED})')
    ap.add_argument('--port', type=int, default=5002,
                    help='화면 포트 (기본 5002 — drive.py 5000, collect.py 5001)')
    ap.add_argument('--jpeg-quality', type=int, default=50)
    ap.add_argument('--log-every', type=int, default=30, help='N프레임마다 상태 출력')
    ap.add_argument('--profile', action='store_true',
                    help='단계별 소요시간(ms) 출력 — 추론이 얼마나 걸리는지 본다')
    ap.add_argument('--replay', metavar='DIR',
                    help='하드웨어 대신 저장 이미지 사용 — 개발 PC에서 확인용')
    ap.add_argument('--replay-loop', action='store_true', help='replay 반복')
    args = ap.parse_args()

    # 하드웨어보다 먼저 만든다. 모델이나 ultralytics 가 없으면 gpio/카메라를
    # 건드리기 전에 끝내야 정리 없이 죽는 일이 없다.
    if args.yolo_remote:
        import yolo_remote
        # 워치독은 주지 않는다 — watch.py 는 수동 주행이라 자동으로 세울 것이 없다
        det = yolo_remote.RemoteDetector(args.yolo_remote, args.yolo_jpeg_quality,
                                         args.yolo_timeout, watchdog_ms=0)
    else:
        det = yolo.Detector(args.model, args.conf, args.imgsz)
        print(f'[yolo] {args.model} imgsz={args.imgsz} conf={args.conf}')
    print(f'[yolo] 클래스 {len(det.names)}종: {", ".join(det.names.values())}')

    if args.replay:
        hw = hardware.ReplayHardware(args.replay, loop=args.replay_loop)
        print(f'[replay] {len(hw.paths)}장 — 하드웨어 없이 확인')
    else:
        hw = hardware.Afb1Hardware()

    man = Manual(hw, args.speed)
    shared = Shared()
    app = make_app(shared, man)

    worker = threading.Thread(target=run_loop,
                              args=(hw, man, det, shared, args,
                                    looplib.Profiler(args.profile)),
                              daemon=True)
    worker.start()

    print()
    print('=' * 52)
    print(f'  화면: http://<Pi주소>:{args.port}')
    print(f'  매 {args.detect_every}프레임마다 탐지합니다')
    print('  방향키(또는 화면 버튼)로 조종하세요 — 누르는 동안만 움직입니다')
    print('  영상이 느리면 DETECT 를 끄거나 --detect-every 를 늘리세요')
    print('=' * 52)
    print('Ctrl+C 로 종료\n')
    try:
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print('\n사용자 종료')
    finally:
        shared.running = False
        worker.join(timeout=2.0)

    hw.shutdown()


if __name__ == '__main__':
    main()
