"""수동 조작 주행 + 카메라 프레임 수집 — 연속 주행 데이터 확보용.

`drive.py` 와 달리 **인지 파이프라인을 전혀 돌리지 않는다.** BEV/이진화/검출/
Pure Pursuit 이 빠지므로 프레임당 비용이 카메라 읽기와 JPEG 저장뿐이다.
사람이 차를 몰고 다니는 동안 프레임만 모으는 것이 이 파일의 전부다.

    afb1.camera.get_image() -> N프레임마다 imwrite
                            -> 웹 스트리밍 + 방향키 조작

하드웨어 접근은 `hardware.py` 를 그대로 쓴다 (Afb1Hardware / ReplayHardware).

사용:
    python3 collect.py                                # 5프레임마다 저장
    python3 collect.py --save-every 3 --speed 50
    python3 collect.py --replay ../project/captures   # 하드웨어 없이 동작 확인

    http://<Pi주소>:5001

조작 (브라우저):
    방향키 또는 화면 버튼 — **누르는 동안만** 이동하고 떼면 중립으로 돌아온다
    REC — 저장 시작/중지. **처음에는 꺼져 있으므로 눌러야 저장이 시작된다**
    비상정지 — 모터 정지 + 서보 중립

저장 형식은 `captures/` 와 같다 (채널 스왑 후 640x480). 즉 `review.py` 와
`evaluate.py` 가 그대로 처리한다.

    python3 tools/review.py --src collected/20260817_101530

안전:
    - 모터는 0, 서보는 중립으로 시작한다. 키를 눌러야만 움직인다
    - 키를 떼거나 브라우저 탭을 벗어나면 즉시 중립으로 돌아간다
    - Ctrl+C / 예외 / 정상 종료 어느 쪽이든 finally 에서 중립 + stop_all()
"""
import argparse
import collections
import os
import threading
import time

import cv2

import config as cfg
import hardware


# ==============================================================================
# 수집 상태
# ==============================================================================
class Collector:
    """수동 조작 명령과 프레임 저장을 맡는다.

    하드웨어는 수집 스레드와 웹 스레드가 함께 만지므로 hw_lock 으로 직렬화한다
    (driver.Driver 가 hw_lock 을 두는 것과 같은 이유다).
    """

    def __init__(self, hw, save_dir, save_every, speed):
        self.hw = hw
        self.save_dir = save_dir
        self.save_every = max(1, save_every)
        self.speed = speed

        self.hw_lock = threading.Lock()
        self.servo_cmd = cfg.SERVO_CENTER
        self.motor_cmd = 0
        # 저장은 꺼진 채로 시작한다. 브라우저를 열고 차를 출발선에 세우는 동안
        # 같은 그림이 쌓이지 않도록, 웹의 REC 버튼을 눌러야 시작한다.
        self.recording = False
        self.frames = 0             # 읽은 프레임
        self.saved = 0              # 저장한 프레임

    # -----------------------------
    # 하드웨어 출력 (양쪽 스레드가 공유)
    # -----------------------------
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
        """방향키 하나의 눌림/뗌을 명령으로 옮긴다.

        떼면 반드시 중립으로 돌린다 — key_up 을 놓치면 차가 계속 달린다.
        """
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

    def store(self, frame):
        """N프레임마다 한 장 저장한다. 저장했으면 True.

        파일명은 저장된 순번이라 번호가 비지 않는다 — 시간축으로 이어 보기 좋다.
        """
        self.frames += 1
        if not self.recording or self.frames % self.save_every:
            return False
        cv2.imwrite(os.path.join(self.save_dir, f'frame_{self.saved:06d}.jpg'), frame)
        self.saved += 1
        return True


class Shared:
    """수집 스레드 <-> 웹 스레드."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
        self.running = True
        self.tel = {'rec': False, 'fps': 0.0, 'frames': 0, 'saved': 0,
                    'servo': cfg.SERVO_CENTER, 'motor': 0}


# ==============================================================================
# 웹 조작 화면
# ==============================================================================
PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<title>Lane Tracer — collect</title>
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
 .rec{background:#1a7f37}.rec.off{background:#8b6b00}
 .estop{background:#b62324}
 #pad{margin-top:14px}
 #pad button{width:74px;height:56px;font-size:22px;background:#30363d}
 .hint{color:#888;font-size:13px;margin-top:8px;line-height:1.5}
</style></head><body>
<h1>Lane Tracer — 수동 주행 + 프레임 수집</h1>
<div class="wrap">
  <img src="/video_feed" width="640">
  <div class="panel">
    <table>
      <tr><td>REC</td><td id="rec">-</td></tr>
      <tr><td>저장</td><td id="saved">-</td></tr>
      <tr><td>읽은 프레임</td><td id="frames">-</td></tr>
      <tr><td>FPS</td><td id="fps">-</td></tr>
      <tr><td>SERVO</td><td id="servo">-</td></tr>
      <tr><td>MOTOR</td><td id="motor">-</td></tr>
    </table>
    <div>
      <button class="rec off" id="b_rec" onclick="toggleRec()">REC</button>
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
        <b>누르는 동안만</b> 움직이고 떼면 멈춥니다.</div>
    </div>
  </div>
</div>
<script>
function post(body){return fetch('/api/control',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
function cmd(a){return post({action:a});}
function keyDown(k){return post({action:'key_down',key:k});}
function keyUp(k){return post({action:'key_up',key:k});}

let rec=false;
function toggleRec(){rec=!rec; post({action:'rec',on:rec});}

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
  for(const k of ['frames','saved','servo','motor'])
    document.getElementById(k).textContent=s[k];
  document.getElementById('fps').textContent=s.fps.toFixed(1);
  document.getElementById('rec').textContent=s.rec?'저장 중':'중지 — REC 를 누르세요';
  rec=s.rec;
  document.getElementById('b_rec').classList.toggle('off',!s.rec);
},300);
</script></body></html>
"""


def make_app(shared, col):
    try:
        from flask import Flask, Response, jsonify, request
    except ImportError as exc:
        raise SystemExit(
            'Flask 가 없어 조작 화면을 띄울 수 없습니다.\n'
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
                col.on_key(data.get('key'), action == 'key_down')

            elif action == 'estop':
                col.neutral()
                print('  [web] EMERGENCY STOP')

            elif action == 'rec':
                col.recording = bool(data.get('on', True))
                print(f"  [web] 저장 {'재개' if col.recording else '중지'}")

        except Exception as exc:                # 요청 하나가 서버를 죽이면 안 된다
            print(f'[web] 명령 처리 실패 ({action}): {exc}')
            return jsonify({'ok': False}), 500

        return jsonify({'ok': True, 'rec': col.recording})

    return app


# ==============================================================================
# 수집 루프
# ==============================================================================
def run_loop(hw, col, shared, args):
    t0 = time.time()
    last_log = 0
    # 순간 FPS 는 최근 N프레임의 실제 간격으로 잰다.
    stamps = collections.deque(maxlen=31)
    pace = (1.0 / args.pace_fps) if args.pace_fps > 0 else 0.0

    try:
        while shared.running:
            loop_start = time.time()

            frame = hw.read()
            if frame is None:
                print('\n더 읽을 프레임이 없습니다')
                break
            # captures/ 와 같은 규격으로 맞춘다 — review.py 가 그대로 읽는다
            frame = cv2.resize(frame, (cfg.W, cfg.H))

            col.store(frame)

            stamps.append(time.time())
            fps = ((len(stamps) - 1) / (stamps[-1] - stamps[0])
                   if len(stamps) > 1 and stamps[-1] > stamps[0] else 0.0)

            tel = {'rec': col.recording, 'fps': fps, 'frames': col.frames,
                   'saved': col.saved, 'servo': col.servo_cmd,
                   'motor': col.motor_cmd}

            # 스트리밍은 저장한 것과 같은 원본 프레임을 쓴다. 화면용 그림을
            # 따로 그리지 않으므로 저장 파일에 주석이 섞일 일이 없다.
            ok, buf = cv2.imencode('.jpg', frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            with shared.lock:
                if ok:
                    shared.jpeg = buf.tobytes()
                shared.tel = tel

            if args.log_every and col.frames - last_log >= args.log_every:
                last_log = col.frames
                print(f'  [{col.frames:5d}] {fps:4.1f}fps  '
                      f"저장 {col.saved:4d}장  {'REC' if col.recording else '---'}  "
                      f'servo={col.servo_cmd:3d} motor={col.motor_cmd:4d}')

            if pace and not args.replay:
                time.sleep(max(0.0, pace - (time.time() - loop_start)))

    except KeyboardInterrupt:
        print('\n사용자 종료')
    except Exception as exc:
        print(f'\n[에러] 수집 루프 예외: {exc}')
    finally:
        shared.running = False
        try:
            col.neutral()
        except Exception:
            pass


def print_summary(col, elapsed):
    print()
    print('=' * 52)
    print(f'  읽은 프레임 {col.frames}장  {col.frames / max(elapsed, 1e-6):.1f} fps')
    print(f'  저장 {col.saved}장 (매 {col.save_every}프레임)')
    print(f'  저장 위치: {col.save_dir}')
    print('=' * 52)
    if col.saved:
        print(f'  확인: python3 tools/review.py --src {col.save_dir}')
    else:
        # 저장이 꺼진 채로 시작하므로 REC 를 안 누르면 한 장도 안 남는다.
        print('  한 장도 저장되지 않았습니다 — 웹에서 REC 를 누르셨나요?')
        try:
            os.rmdir(col.save_dir)      # 빈 폴더는 남기지 않는다
            print(f'  빈 폴더를 지웠습니다: {col.save_dir}')
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(
        description='수동으로 몰면서 카메라 프레임을 N프레임마다 저장한다')
    ap.add_argument('--save-every', type=int, default=5, metavar='N',
                    help='N프레임마다 한 장 저장 (기본 5). 카메라 30fps 기준 '
                         '저장 fps = 30/N')
    ap.add_argument('--out', metavar='DIR',
                    help='저장 폴더. 기본은 collected/<날짜_시각> 으로 매 실행마다 '
                         '새로 만든다 (이전 수집분을 덮어쓰지 않게)')
    ap.add_argument('--speed', type=int, default=cfg.DRIVE_SPEED,
                    help=f'전진/후진 모터 속도 (기본 {cfg.DRIVE_SPEED})')
    ap.add_argument('--port', type=int, default=5001,
                    help='조작 화면 포트 (기본 5001 — drive.py 의 5000 과 겹치지 않게)')
    ap.add_argument('--jpeg-quality', type=int, default=50, help='스트리밍 JPEG 품질')
    ap.add_argument('--log-every', type=int, default=30, help='N프레임마다 상태 출력')
    ap.add_argument('--pace-fps', type=float, default=0.0,
                    help='루프 페이싱 목표 fps. 기본 0 = sleep 없이 카메라 속도에 맡긴다')
    ap.add_argument('--replay', metavar='DIR',
                    help='하드웨어 대신 저장 이미지 사용 — 개발 PC에서 동작 확인용')
    ap.add_argument('--replay-loop', action='store_true', help='replay 반복')
    args = ap.parse_args()

    save_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'collected', time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)

    if args.replay:
        hw = hardware.ReplayHardware(args.replay, loop=args.replay_loop)
        print(f'[replay] {len(hw.paths)}장 — 하드웨어 없이 수집 로직만 확인')
    else:
        hw = hardware.Afb1Hardware()

    col = Collector(hw, save_dir, args.save_every, args.speed)
    shared = Shared()
    app = make_app(shared, col)

    worker = threading.Thread(target=run_loop, args=(hw, col, shared, args),
                              daemon=True)
    t0 = time.time()
    worker.start()

    print()
    print('=' * 52)
    print(f'  조작 화면: http://<Pi주소>:{args.port}')
    print(f'  저장 위치: {save_dir}')
    print(f'  매 {args.save_every}프레임마다 한 장씩 저장합니다')
    print('  방향키(또는 화면 버튼)로 조종하세요 — 누르는 동안만 움직입니다')
    print('  * 저장은 꺼진 채로 시작합니다 — 웹에서 REC 를 눌러야 저장됩니다 *')
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
    print_summary(col, time.time() - t0)


if __name__ == '__main__':
    main()
