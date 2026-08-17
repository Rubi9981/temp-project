"""웹 디버그 대시보드 (자체 Flask 서버).

raspi/L_5_Capture.py 방식을 따른다. afb1.flask 는 쓰지 않는다 — 동작을 확인할 수
없어 화면이 안 뜰 위험이 있고, 직접 띄우면 주소와 포트를 우리가 안다.

    /              대시보드 HTML
    /video_feed    MJPEG 스트림
    /api/status    텔레메트리 JSON
    /api/control   모드 전환 / 비상정지 / 방향키 / 캡처 (POST)

HTML 은 PAGE 상수로 인라인한다 — templates/ 디렉터리를 Pi 로 같이 옮길 필요가 없다.
Flask 는 make_app() 안에서 지연 import 한다. Flask 없는 Pi 에서도 --no-web 경로는
살아 있어야 하므로 이 모듈 자체는 Flask 없이도 import 된다.
"""
import os
import time

import cv2

import config as cfg


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
 .auto{background:#1a7f37}.manual{background:#1f6feb}.stop{background:#8b6b00}
 .estop{background:#b62324}.cap{background:#30363d}
 button.on{outline:3px solid #fff}
 #pad{margin-top:14px}
 #pad button{width:74px;height:56px;font-size:22px;background:#30363d}
 #pad.off{opacity:.35;pointer-events:none}
 .hint{color:#888;font-size:13px;margin-top:8px;line-height:1.5}
</style></head><body>
<h1>Lane Tracer — live debug</h1>
<div class="wrap">
  <img src="/video_feed" width="640">
  <div class="panel">
    <table>
      <tr><td>MODE</td><td id="mode">-</td></tr>
      <tr><td>FPS</td><td id="fps">-</td></tr>
      <tr><td>FPS (평균)</td><td id="fps_avg">-</td></tr>
      <tr><td>SERVO</td><td id="servo">-</td></tr>
      <tr><td>MOTOR</td><td id="motor">-</td></tr>
      <tr><td>STATUS</td><td id="status">-</td></tr>
      <tr><td>FRAMES</td><td id="frames">-</td></tr>
    </table>
    <div>
      <button class="auto"   id="b_AUTO"   onclick="setMode('AUTO')">AUTO</button>
      <button class="manual" id="b_MANUAL" onclick="setMode('MANUAL')">MANUAL</button>
      <button class="stop"   id="b_STOP"   onclick="setMode('STOP')">STOP</button>
    </div>
    <div>
      <button class="estop" onclick="cmd('estop')">EMERGENCY STOP</button>
      <button class="cap"   onclick="cmd('capture')">CAPTURE</button>
    </div>

    <div id="pad" class="off">
      <div style="text-align:center">
        <button data-k="ArrowUp">&#9650;</button></div>
      <div style="text-align:center">
        <button data-k="ArrowLeft">&#9664;</button>
        <button data-k="ArrowDown">&#9660;</button>
        <button data-k="ArrowRight">&#9654;</button></div>
      <div class="hint">MANUAL 모드에서만 동작합니다.<br>
        키보드 방향키로도 조종할 수 있습니다 (누르는 동안만).</div>
    </div>
  </div>
</div>
<script>
function post(body){return fetch('/api/control',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
function cmd(a){return post({action:a});}
function setMode(m){return post({action:'set_mode',mode:m});}
function keyDown(k){return post({action:'key_down',key:k});}
function keyUp(k){return post({action:'key_up',key:k});}

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
  for(const k of ['mode','servo','motor','status','frames'])
    document.getElementById(k).textContent=s[k];
  document.getElementById('fps').textContent=s.fps.toFixed(1);
  document.getElementById('fps_avg').textContent=s.fps_avg.toFixed(1);
  for(const m of ['AUTO','MANUAL','STOP'])
    document.getElementById('b_'+m).classList.toggle('on',s.mode===m);
  document.getElementById('pad').classList.toggle('off',s.mode!=='MANUAL');
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
        data = request.get_json(silent=True) or {}
        action = data.get('action')

        try:
            if action == 'set_mode':
                mode = str(data.get('mode', '')).upper()
                if not driver.set_mode(mode):
                    return jsonify({'ok': False, 'reason': f'unknown mode {mode!r}',
                                    'mode': driver.mode}), 400
                print(f'  [web] 모드 -> {mode}')

            elif action == 'estop':
                driver.set_mode('STOP')
                driver.apply_servo(cfg.SERVO_CENTER)
                print('  [web] EMERGENCY STOP')

            elif action in ('key_down', 'key_up'):
                # 수동 조종은 MANUAL 에서만. 자율주행 중 오조작을 막는다.
                if driver.mode != 'MANUAL':
                    return jsonify({'ok': False, 'reason': 'not MANUAL',
                                    'mode': driver.mode})
                key = data.get('key')
                if action == 'key_down':
                    if key == 'ArrowUp':
                        driver.apply_motor(driver.manual_speed)
                    elif key == 'ArrowDown':
                        driver.apply_motor(-driver.manual_speed)
                    elif key == 'ArrowLeft':
                        driver.apply_servo(cfg.SERVO_MIN)
                    elif key == 'ArrowRight':
                        driver.apply_servo(cfg.SERVO_MAX)
                else:
                    # 키를 떼면 중립으로 — 놓친 key_up 때문에 계속 달리지 않게
                    if key in ('ArrowUp', 'ArrowDown'):
                        driver.apply_motor(0)
                    elif key in ('ArrowLeft', 'ArrowRight'):
                        driver.apply_servo(cfg.SERVO_CENTER)

            elif action == 'capture':
                with shared.lock:
                    raw = None if shared.raw is None else shared.raw.copy()
                if raw is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    path = os.path.join(
                        save_dir, f'cap_{time.strftime("%Y%m%d_%H%M%S")}.jpg')
                    cv2.imwrite(path, raw)
                    print(f'  [web] 캡처 저장: {path}')

        except Exception as exc:                # 웹 요청 하나가 서버를 죽이면 안 된다
            print(f'[web] 명령 처리 실패 ({action}): {exc}')
            return jsonify({'ok': False, 'mode': driver.mode}), 500

        return jsonify({'ok': True, 'mode': driver.mode})

    return app
