import afb1
import cv2
import time
from ultralytics import YOLO

# -----------------------------
# YOLO 모델 로드
# -----------------------------
model = YOLO("best.pt")

# -----------------------------
# 카메라 초기화
# -----------------------------
afb1.camera.init(640, 480, 30)

prev_time = time.time()

# -----------------------------
# 메인 루프
# -----------------------------
while True:

    # 카메라 프레임 가져오기
    frame = afb1.camera.get_image()

    # -----------------------------
    # 채널 문제 해결 (RGBA → BGR)
    # -----------------------------
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # -----------------------------
    # YOLO 추론
    # -----------------------------
    results = model(frame, imgsz=320, conf=0.25)

    # 결과 시각화
    annotated_frame = results[0].plot()

    # -----------------------------
    # FPS 계산
    # -----------------------------
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    cv2.putText(
        annotated_frame,
        f"FPS {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # -----------------------------
    # Flask 스트림용 RGB 변환
    # -----------------------------
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    afb1.flask.imshow("YOLO PT", frame_rgb, 0)