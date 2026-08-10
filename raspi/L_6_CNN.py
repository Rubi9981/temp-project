import afb1
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 모델 로드
model = load_model('CNN.h5')
class_names = ['go', 'left', 'right']

# 시스템 초기화
afb1.camera.init(640, 480, 15)
afb1.gpio.init()

prev_class = None
i = 0

try:
    while True:
        afb1.gpio.motor(60)  # 전진

        frame = afb1.camera.get_image()

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        height = frame_bgr.shape[0]
        roi = frame_bgr[int(height * 0.5):, :]

        input_img = cv2.resize(roi, (64, 64))
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        prediction = model.predict(input_img, verbose=0)
        pred_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        # 콘솔 출력
        print(f"[{i}] {pred_class.upper()} ({confidence:.2f})")
        i += 1

        # 화면 표시용 텍스트 + ROI 박스
        label = f"{pred_class.upper()} ({confidence:.2f})"
        cv2.putText(frame_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame_bgr, (0, int(height * 0.5)), (640, 480), (255, 0, 0), 2)

        # 스트리밍
        afb1.flask.imshow("AFB Camera", frame_bgr, 0)

        # 조향 변경 시만 실행
        if pred_class != prev_class:
            if pred_class == 'left':
                afb1.gpio.servo(30)
            elif pred_class == 'right':
                afb1.gpio.servo(150)
            elif pred_class == 'go':
                afb1.gpio.servo(80)
            prev_class = pred_class

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    afb1.gpio.stop_all()
    afb1.camera.release_camera()
    cv2.destroyAllWindows()
