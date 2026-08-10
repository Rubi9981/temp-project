import cv2
import numpy as np

roi_y_ratio = 0.55                   # ROI 시작 비율
thresh = 127                         # 고정 임계값(160~220 사이로 튜닝)
min_area = 300                       # 컨투어 최소 면적

cap = cv2.VideoCapture('source/lane_test.mp4' )

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    x_center = w // 2

    # -----------------------------
    # ROI 자르기
    # -----------------------------
    y_start = int(h * roi_y_ratio)
    roi = frame[y_start:h, 0:w]

    # -----------------------------
    # 이진화
    # -----------------------------
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 원치 않으면 주석 처리 가능
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    # -----------------------------
    # 컨투어 찾기 / 무게중심 잡기
    # -----------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx_list = []
    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cx_list.append((cx, cy))
        valid_contours.append(cnt)

    cx_list.sort(key=lambda x: x[0])

    # -----------------------------
    # 차선 중심(lane_center_x) 추정
    # -----------------------------
    lane_center_x = None
    left_x = None
    right_x = None

    if len(cx_list) > 0:
        left_x = cx_list[0][0]
        right_x = cx_list[-1][0]

        if len(cx_list) >= 2:
            lane_center_x = int((left_x + right_x) / 2)
        else:
            lane_center_x = left_x

    # -----------------------------
    # 시각화
    # -----------------------------
    vis = frame.copy()

    # ROI 박스
    cv2.rectangle(vis, (0, y_start), (w - 1, h - 1), (255, 255, 0), 2)

    # 카메라 중심선(빨간선)
    cv2.line(vis, (x_center, y_start), (x_center, h), (0, 0, 255), 2)

    # Contour 그리기(ROI -> 원본 좌표로 shift)
    for cnt in valid_contours:
        cnt_shifted = cnt.copy()
        cnt_shifted[:, 0, 1] += y_start
        cv2.drawContours(vis, [cnt_shifted], -1, (0, 255, 0), 2)

    # 중심점들 표시
    for (cx, cy) in cx_list:
        cv2.circle(vis, (cx, cy + y_start), 8, (0, 255, 255), -1)

    if lane_center_x is not None:
        # 차선 중심 점(노란 점 느낌)
        cv2.circle(vis, (lane_center_x, y_start + (h - y_start)//2), 12, (0, 255, 255), -1)

        error = x_center - lane_center_x
        steering = np.clip(error, -200, 200) / 200.0  # -1~+1

        cv2.putText(vis, f"lane_center_x={lane_center_x}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"x_center={x_center}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"error={error}px steering={steering:.2f}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(vis, "Lane not detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Lane Detection (No def)", vis)
    cv2.imshow("Mask (ROI)", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
