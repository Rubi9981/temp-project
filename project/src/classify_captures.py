import cv2
import numpy as np
import os
import shutil

# -----------------------------
# 1) 파라미터 (lane_detector.py 기준)
# -----------------------------
W, H = 640, 480
SRC_POINTS = np.array([
    [36, 262],   # 좌상단
    [588, 269],  # 우상단
    [605, 406],  # 우하단
    [17, 413]    # 좌하단
], dtype=np.float32)

ROI_Y_RATIO = 0.55
MIN_AREA = 300
BLUR_KSIZE = (5, 5)
MORPH_KSIZE = (5, 5)
MORPH_ITER_OPEN = 1
MORPH_ITER_CLOSE = 2
USE_ADAPTIVE = True
ADAPTIVE_METHOD = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
ADAPTIVE_TYPE = cv2.THRESH_BINARY
BLOCK_SIZE = 21
C_VALUE = -4

# 분류 임계값 설정 (픽셀 오차 기준)
# error = x_center - center_pt_warp[0]
# error > ERROR_THRESHOLD: 좌회전 (left)
# error < -ERROR_THRESHOLD: 우회전 (right)
# 그 외: 직진 (go)
ERROR_THRESHOLD = 20

CAPTURES_DIR = '/Users/rubi/Desktop/myfolder/MJY/lane_tracer/project/captures'
CATEGORIES = ['go', 'left', 'right', 'undefined']

# -----------------------------
# 2) 원근 변환 함수
# -----------------------------
def warp_image(image, src_pts, width=W, height=H):
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_points)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped

# -----------------------------
# 3) 프레임 분류 함수
# -----------------------------
def classify_frame(frame):
    frame = cv2.resize(frame, (W, H))
    warp_frame = warp_image(frame, SRC_POINTS, W, H)

    h, w = warp_frame.shape[:2]
    x_center = w // 2

    y_start = int(h * ROI_Y_RATIO)
    roi = warp_frame[y_start:h, 0:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if BLUR_KSIZE is not None:
        gray = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)

    if USE_ADAPTIVE:
        cur_blockSize = BLOCK_SIZE if BLOCK_SIZE % 2 != 0 else BLOCK_SIZE + 1
        mask = cv2.adaptiveThreshold(gray, 255, ADAPTIVE_METHOD, ADAPTIVE_TYPE, cur_blockSize, C_VALUE)
    else:
        _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)
    if MORPH_ITER_OPEN > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITER_OPEN)
    if MORPH_ITER_CLOSE > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITER_CLOSE)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        points.append((cx, cy))

    if not points:
        return 'undefined'

    points.sort(key=lambda p: p[0])
    left_roi = points[0]
    right_roi = points[-1]

    center_x = int((left_roi[0] + right_roi[0]) / 2)
    error = x_center - center_x

    if error > ERROR_THRESHOLD:
        return 'left'
    elif error < -ERROR_THRESHOLD:
        return 'right'
    else:
        return 'go'

# -----------------------------
# 4) 분류 실행 메인 함수
# -----------------------------
def main():
    # 카테고리 폴더 생성
    for cat in CATEGORIES:
        os.makedirs(os.path.join(CAPTURES_DIR, cat), exist_ok=True)

    files = [f for f in os.listdir(CAPTURES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"총 {len(files)}개 이미지 파일 분류를 시작합니다...")

    stats = {'go': 0, 'left': 0, 'right': 0, 'undefined': 0}

    for filename in files:
        file_path = os.path.join(CAPTURES_DIR, filename)
        if not os.path.isfile(file_path):
            continue

        frame = cv2.imread(file_path)
        if frame is None:
            continue

        cat = classify_frame(frame)
        dest_path = os.path.join(CAPTURES_DIR, cat, filename)
        shutil.move(file_path, dest_path)
        stats[cat] += 1

    print("\n✅ 분류 완료 결과:")
    for cat, count in stats.items():
        print(f"  - {cat}: {count}개")

if __name__ == '__main__':
    main()
