# 1. 라이브러리 불러오기
import cv2
import numpy as np

# 2. image 배열 설정하기 - 내가 원하는 이미지 리스트 구성
# image 변수에 4차원 리스트로 색상(B, G, R) 정보를 저장
image = cv2.imread('source/flower.jpg')

# 3. Grayscale 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 4. Binarization
_, binary = cv2.threshold(gray, 177, 255, cv2.THRESH_BINARY)

# 4. 결과 출력
cv2.imshow("Original Image", gray)
cv2.imshow("Grayscale Image", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()