import cv2
import numpy as np

# 이미지 읽기 및 전처리
img = cv2.imread('source/shape_rect.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

# 컨투어 찾기
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

# 사각형
x, y, w, h = cv2.boundingRect(contour)
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 최소 영역 사각형
rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rect)
box = np.int64(box)
cv2.drawContours(img, [box], 0, (255, 0, 255), 2)

# 원
center, radius = cv2.minEnclosingCircle(contour)
center = (int(center[0]), int(center[1]))
radius = int(radius)
cv2.circle(img, center, radius, (255, 0, 0), 2)

# 타원
ellipse = cv2.fitEllipse(contour)
cv2.ellipse(img, ellipse, (0, 255, 255), 2)

# 직선
[vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
vx, vy, x, y = vx[0], vy[0], x[0], y[0]

lefty = int((-x * vy / vx) + y)
righty = int(((img.shape[1] - x) * vy / vx) + y)
cv2.line(img, (img.shape[1] - 1, righty), (0, lefty), (0, 0, 255), 2)

# 결과 이미지 출력
cv2.imshow('Bound Shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
