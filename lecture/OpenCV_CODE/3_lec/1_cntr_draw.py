import cv2
import numpy as np

# 이미지 읽기 및 전처리
#img = cv2.imread('source/shape_rect.jpg')
img = cv2.imread('source/cropped.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
# 컨투어 찾기
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 컨투어 그리기
cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

# 결과 이미지 출력
cv2.imshow('Binary', thresh)
cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()