import cv2
import numpy as np

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # HSV 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 빨간색 HSV 범위
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # 빨간색 마스크
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 컨투어 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 결과 그릴 프레임
    contour_image = frame.copy()

    # 컨투어가 있을 때만 처리
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
    
        area = cv2.contourArea(contours)
        if area > 500:  # 노이즈 제거용 임계값
            # 컨투어
            cv2.drawContours(contour_image, [contours], -1, (0, 255, 0), 2)

            # 바운딩 박스
            x, y, w, h = cv2.boundingRect(contours)
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.putText(
                contour_image,
                "RED",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    # 결과 출력
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Red Mask', mask)
    cv2.imshow('Red Detection', contour_image)

    # 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
cap.release()
cv2.destroyAllWindows()
