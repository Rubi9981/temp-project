import cv2
import numpy as np

# coordinate.py와 동일한 이미지 경로 및 규격 설정
W, H = 640, 480
IMG_PATH = 'captures/frame_20260803_162123.jpg'


### 1단계 : 원근 변환 함수 정의 (Bird's Eye View로 변환하기 위한 함수) ###
def warp_image(image, src_points):
    # 출력 이미지의 너비와 높이 (고정)
    width, height = W, H

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


### 2단계 : 원본 이미지에서 기준이 되는 원근 변환 좌표 정의 ###
# coordinate.py에서 추출한 좌표 (좌상 → 우상 → 우하 → 좌하)
src_points = np.array([
    [36, 262],   # 좌상단
    [588, 269],  # 우상단
    [605, 406],  # 우하단 (x: 605, y: 406)
    [17, 413]    # 좌하단 (x: 17, y: 413)
], dtype=np.float32)


### 3단계 : 이미지 파일 불러오기 및 리사이즈 ###
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {IMG_PATH}")

img = cv2.resize(img, (W, H))


### 4단계 : Bird's Eye View 변환 적용 ###
warped_img = warp_image(img, src_points)

# 원본 이미지에 변환 영역(다각형) 표시
preview_img = img.copy()
cv2.polylines(preview_img, [src_points.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)


### 5단계 : 결과 시각화 ###
print("결과 창을 닫거나 아무 키를 누르면 종료됩니다.")
cv2.imshow('Original Image with ROI', preview_img)   # 원본 이미지 (영역 표시)
cv2.imshow('Warped (Bird\'s Eye View)', warped_img)  # BEV 변환 이미지


### 6단계 : 대기 및 창 닫기 ###
cv2.waitKey(0)
cv2.destroyAllWindows()
