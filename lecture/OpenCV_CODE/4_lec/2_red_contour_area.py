import cv2
import numpy as np

# 이미지 로드
image = cv2.imread('source/traffic.jpg')

# 이미지를 HSV 색 공간으로 변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 빨간색의 HSV 범위 정의
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# 빨간색 영역 마스크 생성
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_hsv = cv2.bitwise_or(mask1, mask2)

# 컨투어 검출
contours, hierarchy = cv2.findContours(
    mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# 컨투어 + 바운딩 박스 그리기
for cnt in contours:
    area = cv2.contourArea(cnt)

    # 너무 작은 노이즈 제거
    if area > 10000:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "RED", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# 결과 이미지 표시
cv2.imshow('Original Image', image)
cv2.imshow('Red Mask', mask_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()
