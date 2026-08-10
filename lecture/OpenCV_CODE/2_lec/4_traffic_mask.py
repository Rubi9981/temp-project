import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
image = cv2.imread('source/traffic.jpg')

# 이미지를 HSV 색 공간으로 변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 빨간색 HSV 범위 정의
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([5, 255, 255])
lower_red2 = np.array([175, 120, 70])
upper_red2 = np.array([180, 255, 255])

# 빨간색 영역 마스크 생성
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

# 노란색 HSV 범위 정의
lower_yellow = np.array([23, 170, 170])
upper_yellow = np.array([33, 255, 255])

# 노란색 영역 마스크 생성
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

# 초록색 HSV 범위 정의
lower_green = np.array([40, 120, 120])
upper_green = np.array([80, 255, 255])

# 초록색 영역 마스크 생성
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# 마스크 합치기 (빨강 + 노랑 + 초록)
mask_all = cv2.bitwise_or(mask_red, mask_yellow)
mask_all = cv2.bitwise_or(mask_all, mask_green)

# 원본 이미지와 각 마스크 합성
result_red = cv2.bitwise_and(image, image, mask=mask_red)
result_yellow = cv2.bitwise_and(image, image, mask=mask_yellow)
result_green = cv2.bitwise_and(image, image, mask=mask_green)
result_all = cv2.bitwise_and(image, image, mask=mask_all)

# OpenCV로 결과 출력
# ==============================
cv2.imshow('Original Image', image)
cv2.imshow('Red Mask', mask_red)
cv2.imshow('Yellow Mask', mask_yellow)
cv2.imshow('Green Mask', mask_green)
cv2.imshow('Detected All Colors', result_all)
cv2.waitKey(0)
cv2.destroyAllWindows()

