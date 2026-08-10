# 1. 라이브러리 불러오기
import cv2
import numpy as np

# 2. image 배열 설정하기 - 내가 원하는 이미지 리스트 구성
# image 변수에 4차원 리스트로 색상(B, G, R) 정보를 저장
image = cv2.imread('source/flower.jpg')

# 3. 이미지 출력 및 창 설정 - 결과 화면 띄우기
print(image.shape)
cv2.imshow("image", image)                   # 이미지 띄우기
cv2.waitKey(0)                                # 키보드 입력 대기
cv2.destroyAllWindows()                       # 모든 창 닫기
