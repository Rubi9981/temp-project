import cv2
import numpy as np

# 1. BGR 이미지 직접 생성
image = [[[[0, 0, 255], [0, 255, 0]],
          [[255, 0, 0], [0, 255, 255]]]]

image = np.array(image, dtype=np.uint8)
image = image[0]  # (2, 2, 3)

# 2. BGR → HSV 변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print(hsv)
image = cv2.resize(image, (480, 480),interpolation=cv2.INTER_NEAREST)
hsv = cv2.resize(hsv, (480, 480),interpolation=cv2.INTER_NEAREST)
# 3. OpenCV로 출력
cv2.imshow("BGR Image", image)
cv2.imshow("HSV Image", hsv)

h, s, v = cv2.split(hsv)

cv2.imshow("H channel", h)
cv2.imshow("S channel", s)
cv2.imshow("V channel", v)

cv2.waitKey(0)
cv2.destroyAllWindows()
