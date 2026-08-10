import cv2
import numpy as np

### 1단계 : 원근 변환 함수 정의 (Bird's Eye View로 변환하기 위한 함수) ###
def warp_image(image, src_points):
    # 출력 이미지의 너비와 높이 (고정)
    width, height = 640, 480

    # 변환 후 목적지 좌표 (좌상 → 우상 → 우하 → 좌하)
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # 원근 변환 행렬 계산
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 변환 적용
    warped_image = cv2.warpPerspective(image, matrix, (width, height))

    return warped_image



### 2단계 : 원본 영상에서 기준이 되는 원근 변환 좌표 정의 ###
# → 실제 도로 영상이라면 눈으로 확인 후 클릭해서 추출한 값이어야 함
# → 영상 크기가 640x480일 때 기준
src_points = np.array([
    [210, 220],  # 좌상단: 왼쪽 차선 시작점
    [410, 220],  # 우상단: 오른쪽 차선 시작점
    [639, 400],  # 우하단: 오른쪽 바닥 가까운 지점
    [0, 400]     # 좌하단: 왼쪽 바닥 가까운 지점
], dtype=np.float32)



### 3단계 : 영상 파일 불러오기 ###
cap = cv2.VideoCapture('source/lane_test.mp4')


### 4단계 : 프레임 반복 처리 (영상 재생 루프) ###
while True:
    # 프레임 읽기
    ret, frame = cap.read()

    # 프레임을 640x480으로 리사이즈 (화면에 꽉 차도록 맞춤)
    frame = cv2.resize(frame, (640, 480))

    ### 5단계 : Bird's Eye View 변환 적용 ###
    warped_frame = warp_image(frame, src_points)


    ### 6단계 : 결과 시각화 ###
    cv2.imshow('Original Frame', frame)           # 원본 영상
    cv2.imshow('Warped (Bird\'s Eye View)', warped_frame)  # 변환된 영상


    ### 7단계 : 종료 조건 처리 ###
    # 'q' 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


### 8단계 : 리소스 해제 및 창 닫기 ###
cv2.destroyAllWindows()
