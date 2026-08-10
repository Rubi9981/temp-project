import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
image = cv2.imread('source/traffic.jpg')

# 이미지를 HSV 색 공간으로 변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 빨간색의 HSV 범위 정의
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# 빨간색 영역을 마스크로 만들기
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_hsv = cv2.bitwise_or(mask1, mask2)

# 원본 이미지와 마스크를 합성
result_hsv = cv2.bitwise_and(image, image, mask=mask_hsv)

# 결과 이미지 표시
cv2.imshow('Original Image', image)
cv2.imshow('Red Mask', mask_hsv)
cv2.imshow('Detected Red Color', result_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()

#matplotlib로 표시
#matplotlib는 RGB로 표현하기에 이미지를 RGB로 변환하여야 한다.
image_mat2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask_mat2 = cv2.cvtColor(mask_hsv, cv2.COLOR_BGR2RGB)
result_mat2 = cv2.cvtColor(result_hsv, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_mat2)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask_mat2, cmap='gray')
plt.title("Red Mask")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(result_mat2)
plt.title("Detected Red Color")
plt.axis('off')

plt.tight_layout()
plt.show()