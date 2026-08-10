import cv2
import numpy as np

# ROI 영역 정의 (4개 점)
roi_points = np.array([
    [100, 100],
    [540, 100],
    [540, 380],
    [100, 380]
], dtype=np.int32)  # 원하는 좌표로 수정 가능

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()

    # 빈 마스크 생성 (frame과 동일한 크기, 흑백)
    roi_mask = np.zeros((480, 640), dtype=np.uint8)

    # ROI 영역만 흰색(255)으로 채움
    cv2.fillPoly(roi_mask, [roi_points], 255)
    # HSV 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 빨간색 마스크 생성
    lower_red1 = np.array([0, 60, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 60, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # ROI 영역에만 마스크 적용
    red_roi_mask = cv2.bitwise_and(red_mask, roi_mask)

    # 컨투어 검출
    contours, _ = cv2.findContours(red_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Red Object", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 시각화
    result = cv2.bitwise_and(frame, frame, mask=red_roi_mask)
    cv2.polylines(frame, [roi_points], isClosed=True, color=(255, 255, 0), thickness=2)

    cv2.imshow('Original Frame with ROI', frame)
    cv2.imshow('Red Mask (ROI only)', red_roi_mask)
    cv2.imshow('Detected Red Color (ROI only)', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
