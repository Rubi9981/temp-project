import cv2
import numpy as np

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 노이즈 제거용 임계값 (환경에 따라 조절)
MIN_AREA = 500

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # HSV 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ==============================
    # 빨강 / 노랑 / 초록 HSV 범위 정의
    # ==============================
    lower_red1 = np.array([0, 120, 120])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 120])
    upper_red2 = np.array([180, 255, 255])

    lower_yellow = np.array([25, 50, 50])
    upper_yellow = np.array([45, 255, 255])

    lower_green = np.array([40, 70, 70])
    upper_green = np.array([80, 255, 255])

    # ==============================
    # 각 색 마스크 생성
    # ==============================
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # (선택) 전체 마스크
    mask_all = cv2.bitwise_or(mask_red, mask_yellow)
    mask_all = cv2.bitwise_or(mask_all, mask_green)

    # 결과 그릴 프레임
    contour_image = frame.copy()

    # ==============================
    # 색상별 컨투어 전부 처리 함수
    # ==============================
    def draw_contours(mask, label, color):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # 면적 필터링 (작은 노이즈 제거)
            if area > MIN_AREA:
                # 바운딩 박스
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(contour_image, (x, y), (x + w, y + h), color, 2)

                # 텍스트
                cv2.putText(
                    contour_image,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

    # ==============================
    # 빨/노/초 각각 적용 (max 없음)
    # ==============================
    draw_contours(mask_red, "RED", (0, 0, 255))
    draw_contours(mask_yellow, "YELLOW", (0, 255, 255))
    draw_contours(mask_green, "GREEN", (0, 255, 0))

    # 결과 출력
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Mask (All)', mask_all)
    cv2.imshow('Detection', contour_image)

    # 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
cap.release()
cv2.destroyAllWindows()
