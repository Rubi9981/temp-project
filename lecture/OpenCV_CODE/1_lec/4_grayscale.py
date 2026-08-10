# # 1. 라이브러리 불러오기
# import cv2
# import numpy as np

# # 2. image 배열 설정하기 - 내가 원하는 이미지 리스트 구성
# # image 변수에 4차원 리스트로 색상(B, G, R) 정보를 저장
# image = cv2.imread('source/flower.jpg')

# # 3. Grayscale 변환
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # 4. 결과 출력
# cv2.imshow("Original Image", image)
# cv2.imshow("Grayscale Image", gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 1. 라이브러리 불러오기
import cv2
import numpy as np

# 2. image 배열 설정하기 - 내가 원하는 이미지 리스트 구성
# image 변수에 4차원 리스트로 색상(B, G, R) 정보를 저장
image = cv2.imread('source/flower.jpg')

# 3. Grayscale 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 4. 원본과 Grayscale 이미지 나란히 붙이기
# grayscale 이미지는 1채널이므로 3채널로 변환 - hstack은 같은 채널끼리만 가능하기 때문!
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

combined = np.hstack((image, gray_bgr))

# 5. 결과 출력
cv2.imshow("Original | Grayscale", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()