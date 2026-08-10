# 1. 라이브러리 불러오기
import cv2
import numpy as np

# 2. image 배열 설정하기 - 내가 원하는 이미지 리스트 구성
# image 변수에 4차원 리스트로 색상(B, G, R) 정보를 저장
image = [[[[255, 0, 255], [0, 255, 0]],
          [[255, 0, 0], [0, 255, 255]]]]

# 3. numpy 배열로 변환 - OpenCV에서 인식 가능한 형태로 만들기
image = np.array(image, dtype=np.uint8)

# 4. 배열 크기(shape) 확인하기 - 이미지 데이터 구조 확인
print("image.shape:", image.shape)  # 출력 결과: (1, 2, 2, 3) (이미지 개수, 세로, 가로, 색상)

# 5. 배열 차원 수정 - 실제 이미지 데이터만 추출 - 배치 이미지 하나만 꺼내기
image = image[0]  # (2, 2, 3) 형태로 변환

# 6. 이미지 출력 및 창 설정 - 결과 화면 띄우기
cv2.namedWindow("image", cv2.WINDOW_NORMAL)  # 창 크기 자유 조절 가능
cv2.imshow("image", image)                   # 이미지 띄우기
cv2.waitKey(0)                                # 키보드 입력 대기
cv2.destroyAllWindows()                       # 모든 창 닫기
