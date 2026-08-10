import cv2
import numpy as np

# 1. 이미지 로드
image = cv2.imread("source/traffic2.jpg")
if image is None:
    raise FileNotFoundError("이미지를 불러올 수 없습니다.")

height, width = image.shape[:2]

# 2. ROI 다각형 좌표 정의 (시계 방향)
# 예: 화면 중앙 상단 영역
roi_points = np.array([
    [width * 0.3, height * 0.2],
    [width * 0.7, height * 0.2],
    [width * 0.8, height * 0.6],
    [width * 0.2, height * 0.6]
], dtype=np.int32)

# 3. 빈 마스크 생성 (1채널)
mask = np.zeros((height, width), dtype=np.uint8)

# 4. 다각형 ROI 영역 채우기
cv2.fillPoly(mask, [roi_points], 255)

# 5. ROI 적용 (원본 이미지 유지)
roi_result = cv2.bitwise_and(image, image, mask=mask)

# 6. 결과 출력
cv2.imshow("Original Image", image)
cv2.imshow("Polygon ROI Mask", mask)
cv2.imshow("ROI Applied Result", roi_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
