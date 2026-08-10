import os
import sys
import time
import threading
import cv2
import numpy as np
from flask import Flask, render_template, request, Response, jsonify

# -------------------------------------------------------------
# 1. 하드웨어 라이브러리(afb1) 및 머신러닝 모듈 로드 (안전 가드 포함)
# -------------------------------------------------------------
try:
    import afb1
    IS_RASPI = True
except ImportError:
    # PC 환경 등에서 테스트할 경우를 위한 Mock 클래스
    print("[경고] 'afb1' 모듈을 찾을 수 없어 가상 모드(Mock Mode)로 동작합니다.")
    IS_RASPI = False
    class MockAFB1:
        class camera:
            @staticmethod
            def init(w, h, fps): pass
            @staticmethod
            def get_image():
                # 더미 카메라 프레임 생성 (테스트용)
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(dummy, "AFB1 Mock Camera", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return dummy
            @staticmethod
            def release_camera(): pass
        class gpio:
            @staticmethod
            def init(): pass
            @staticmethod
            def stby(val): pass
            @staticmethod
            def motor(speed, *args): pass
            @staticmethod
            def servo(angle): pass
            @staticmethod
            def led(l, r): pass
            @staticmethod
            def stop_all(): pass
        class flask:
            @staticmethod
            def imshow(name, frame, delay): pass
    afb1 = MockAFB1()

# CNN 모델 로드 시도
CNN_MODEL = None
CLASS_NAMES = ['go', 'left', 'right']
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'CNN.h5')

if os.path.exists(MODEL_PATH):
    try:
        from tensorflow.keras.models import load_model
        CNN_MODEL = load_model(MODEL_PATH)
        print(f"[정보] 자율주행 CNN 모델 '{MODEL_PATH}' 로드 완료.")
    except Exception as e:
        print(f"[경고] CNN 모델 로드 중 오류 발생: {e}")
else:
    print(f"[알림] '{MODEL_PATH}' 모델 파일이 없습니다. 기본 룰/수동 주행 모드로 대기합니다.")

# -------------------------------------------------------------
# 2. 전역 상태 및 파라미터 정의
# -------------------------------------------------------------
app = Flask(__name__)

SAVE_DIR = os.path.join(os.path.dirname(__file__), 'captures')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 공유 상태 변수
state = {
    "mode": "MANUAL",        # "AUTO", "MANUAL", "STOP"
    "fps": 0.0,
    "servo_angle": 90,       # 30 (좌회전) ~ 90 (직진) ~ 150 (우회전)
    "motor_speed": 0,        # -100 ~ +100 (또는 -255 ~ 255)
    "prediction": "IDLE",    # 최근 AI 판단 결과
    "is_running": True
}

# 영상 버퍼 (스레드 간 공유)
latest_frame = None
latest_jpeg = None
lock = threading.Lock()

# -------------------------------------------------------------
# 3. 자율주행 & 카메라 캡처 워커 스레드
# -------------------------------------------------------------
def drive_worker():
    global latest_frame, latest_jpeg, state

    # 카메라 및 GPIO 초기화
    afb1.camera.init(640, 480, 30)
    afb1.gpio.init()
    if hasattr(afb1.gpio, 'stby'):
        afb1.gpio.stby(1)

    prev_time = time.time()
    prev_class = None

    print("[정보] 카메라 및 자율주행 제어 루프가 시작되었습니다.")

    try:
        while state["is_running"]:
            loop_start = time.time()

            # 1) 카메라 원본 프레임 획득
            frame = afb1.camera.get_image()
            if frame is None:
                time.sleep(0.01)
                continue

            # 채널 규격 정규화 (RGBA -> BGR 또는 RGB)
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif not IS_RASPI:
                frame_bgr = frame.copy()
            else:
                # afb1.camera.get_image()는 기본적으로 RGB/BGR 포맷 반환
                # 주행 모델 처리를 위해 BGR 기준 복사
                frame_bgr = frame.copy()

            h, w = frame_bgr.shape[:2]
            pred_label = "MANUAL"

            # 2) 주행 모드에 따른 로직 수행
            if state["mode"] == "AUTO":
                # 기본 전진 속도 설정 (환경에 맞춰 50~70 조절)
                current_motor = 60
                state["motor_speed"] = current_motor
                afb1.gpio.motor(current_motor)

                # 하단 50% 관심 영역(ROI) 추출
                roi = frame_bgr[int(h * 0.5):, :]

                if CNN_MODEL is not None:
                    # CNN 전처리 및 추론
                    input_img = cv2.resize(roi, (64, 64))
                    input_img = input_img.astype(np.float32) / 255.0
                    input_img = np.expand_dims(input_img, axis=0)

                    prediction = CNN_MODEL.predict(input_img, verbose=0)
                    pred_idx = np.argmax(prediction)
                    pred_class = CLASS_NAMES[pred_idx]
                    confidence = float(np.max(prediction))

                    pred_label = f"{pred_class.upper()} ({confidence:.2f})"
                    state["prediction"] = pred_label

                    # 조향 제어 (변화가 있을 때만 서보 변경)
                    if pred_class != prev_class:
                        if pred_class == 'left':
                            state["servo_angle"] = 40
                            afb1.gpio.servo(40)
                        elif pred_class == 'right':
                            state["servo_angle"] = 140
                            afb1.gpio.servo(140)
                        elif pred_class == 'go':
                            state["servo_angle"] = 80
                            afb1.gpio.servo(80)
                        prev_class = pred_class
                else:
                    # 모델이 없을 경우 기본 직진
                    pred_label = "AUTO (NO MODEL - FORWARD)"
                    state["prediction"] = pred_label
                    state["servo_angle"] = 80
                    afb1.gpio.servo(80)

            elif state["mode"] == "STOP":
                afb1.gpio.motor(0)
                state["motor_speed"] = 0
                state["prediction"] = "EMERGENCY STOP"

            # 3) HUD (Head-Up Display) 오버레이 렌더링
            vis_frame = frame_bgr.copy()

            # ROI 가이드라인 표시
            cv2.rectangle(vis_frame, (0, int(h * 0.5)), (w, h), (255, 120, 0), 2)

            # 상단 상태 텍스트
            mode_color = (0, 255, 0) if state["mode"] == "AUTO" else ((0, 200, 255) if state["mode"] == "MANUAL" else (0, 0, 255))
            cv2.putText(vis_frame, f"MODE: {state['mode']}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, mode_color, 2)
            cv2.putText(vis_frame, f"FPS: {state['fps']:.1f}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
            cv2.putText(vis_frame, f"SERVO: {state['servo_angle']} deg", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
            cv2.putText(vis_frame, f"MOTOR: {state['motor_speed']}", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

            # AI 예측 결과 중앙 상단
            cv2.putText(vis_frame, f"PRED: {state['prediction']}", (w - 320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 하단 조향 각도 게이지 바
            center_x = w // 2
            gauge_y = h - 25
            cv2.line(vis_frame, (center_x - 100, gauge_y), (center_x + 100, gauge_y), (100, 100, 100), 3)
            # 30~150도를 게이지 위치(-100 ~ +100)로 매핑
            offset = int((state["servo_angle"] - 90) * (100 / 60))
            cv2.circle(vis_frame, (center_x + offset, gauge_y), 8, (0, 255, 0), -1)

            # 4) FPS 계산
            now = time.time()
            dt = now - prev_time
            if dt > 0:
                state["fps"] = 1.0 / dt
            prev_time = now

            # 5) JPEG 인코딩 및 최신 버퍼 저장 (웹 스트리밍용)
            # JPEG 인코딩 품질 75로 설정하여 대역폭 절약 및 초저지연 유지
            _, jpeg_buffer = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpeg_bytes = jpeg_buffer.tobytes()

            with lock:
                latest_frame = vis_frame.copy()
                latest_jpeg = jpeg_bytes

            # 루프 간 짧은 휴식 (카메라 프레임 레이트에 맞춤)
            elapsed = time.time() - loop_start
            sleep_time = max(0.005, (1.0 / 30.0) - elapsed)
            time.sleep(sleep_time)

    except Exception as e:
        print(f"[에러] 주행 스레드 예외 발생: {e}")
    finally:
        print("[안내] 주행 스레드 정지 및 모터 전체 정지")
        afb1.gpio.stop_all()
        if hasattr(afb1.camera, 'release_camera'):
            afb1.camera.release_camera()

# -------------------------------------------------------------
# 4. Flask 웹 라우트 및 API 엔드포인트
# -------------------------------------------------------------
@app.route('/')
def index():
    return render_template('stream.html')

def mjpeg_generator():
    """웹 브라우저로 최신 프레임을 지속적으로 밀어주는 제너레이터"""
    while state["is_running"]:
        with lock:
            frame_data = latest_jpeg

        if frame_data is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.033)  # 약 30 FPS 스트리밍

@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "mode": state["mode"],
        "fps": state["fps"],
        "servo_angle": state["servo_angle"],
        "motor_speed": state["motor_speed"],
        "prediction": state["prediction"]
    })

@app.route('/api/control', methods=['POST'])
def control():
    global state
    data = request.get_json() or {}
    action = data.get('action')

    if action == 'set_mode':
        new_mode = data.get('mode', 'MANUAL').upper()
        if new_mode in ['AUTO', 'MANUAL', 'STOP']:
            state['mode'] = new_mode
            if new_mode == 'STOP':
                afb1.gpio.motor(0)
                state['motor_speed'] = 0

    elif action == 'emergency_stop':
        state['mode'] = 'STOP'
        afb1.gpio.motor(0)
        state['motor_speed'] = 0
        state['servo_angle'] = 90
        afb1.gpio.servo(90)

    elif action == 'capture':
        with lock:
            if latest_frame is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(SAVE_DIR, f"cap_{timestamp}.jpg")
                cv2.imwrite(save_path, latest_frame)
                print(f"[캡처] 저장 완료: {save_path}")

    elif action == 'key_down' and state['mode'] == 'MANUAL':
        key = data.get('key')
        if key == 'ArrowUp':
            state['motor_speed'] = 70
            afb1.gpio.motor(70)
        elif key == 'ArrowDown':
            state['motor_speed'] = -70
            afb1.gpio.motor(-70)
        elif key == 'ArrowLeft':
            state['servo_angle'] = 40
            afb1.gpio.servo(40)
        elif key == 'ArrowRight':
            state['servo_angle'] = 140
            afb1.gpio.servo(140)

    elif action == 'key_up' and state['mode'] == 'MANUAL':
        key = data.get('key')
        if key in ['ArrowUp', 'ArrowDown']:
            state['motor_speed'] = 0
            afb1.gpio.motor(0)
        elif key in ['ArrowLeft', 'ArrowRight']:
            state['servo_angle'] = 90
            afb1.gpio.servo(90)

    return jsonify({"status": "ok", "current_mode": state["mode"]})

# -------------------------------------------------------------
# 5. 메인 실행 진입점
# -------------------------------------------------------------
if __name__ == '__main__':
    # 백그라운드 주행 & 영상 캡처 스레드 구동
    worker_t = threading.Thread(target=drive_worker, daemon=True)
    worker_t.start()

    print("\n=======================================================")
    print(" 🚀 AFB-1 자율주행 & 실시간 스트리밍 서버가 시작되었습니다.")
    print(" 📡 내 컴퓨터(PC) 웹 브라우저에서 아래 주소로 접속하세요:")
    print("    👉 http://<라즈베리파이_IP_주소>:5000")
    print("=======================================================\n")

    try:
        # Flask 웹 서버 실행 (0.0.0.0으로 외부 PC 접속 허용)
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[안내] 사용자에 의해 서버가 종료됩니다.")
    finally:
        state["is_running"] = False
        afb1.gpio.stop_all()
        print("[완료] 모든 하드웨어가 안전하게 정지되었습니다.")
