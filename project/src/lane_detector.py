import cv2
import numpy as np
import os

# ==============================================================================
# [설정 및 파라미터 튜닝] - 자주 수정하는 파라미터 모음
# ==============================================================================

# 1. 파일 경로 설정
VIDEO_PATH = 'source/lane_test.mp4'
IMAGE_PATH = 'captures/3left.jpg'

# 2. 이미지 규격 및 BEV (Warp) 기준 좌표 ([좌상, 우상, 우하, 좌하] 순서)
W, H = 640, 480
SRC_POINTS = np.array([
    [36, 262],   # 좌상단: 왼쪽 차선 원거리 지점
    [588, 269],  # 우상단: 오른쪽 차선 원거리 지점
    [605, 406],  # 우하단: 오른쪽 차선 근거리 지점
    [17, 413]    # 좌하단: 왼쪽 차선 근거리 지점
], dtype=np.float32)

# 3. 차선 검출 영역 및 면적 파라미터
ROI_Y_RATIO = 0.55     # BEV 이미지 하단 ROI 시작 비율 (0.55 = 하단 45% 영역 사용)
MIN_AREA = 300          # 노이즈 제거용 최소 차선 컨투어 면적

# 4. 블러 / 모폴로지 커널 설정
BLUR_KSIZE = (5, 5)
MORPH_KSIZE = (5, 5)
MORPH_ITER_OPEN = 1
MORPH_ITER_CLOSE = 2

# 5. 이진화 설정 (True: 적응형 이진화 사용, False: 고정 임계값 사용)
USE_ADAPTIVE = True

# (고정 임계값 모드: USE_ADAPTIVE = False 일 때 적용)
THRESH = 180            # 흰색 차선의 경우 180~200 사이 권장

# (적응형 이진화 모드: USE_ADAPTIVE = True 일 때 적용)
ADAPTIVE_METHOD = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
ADAPTIVE_TYPE = cv2.THRESH_BINARY
BLOCK_SIZE = 21         # 반드시 홀수 (11, 15, 21, 31 ...)
C_VALUE = -4            # 감도 조절 (차선 검출이 약하면 더 낮은 음수, 노이즈가 많으면 높임)

# ==============================================================================


# -----------------------------
# 1) Bird's Eye View (Warp) 함수 & 좌표 역투영 함수
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
    return warped, matrix


def project_point(pt_xy, matrix):
    pt = np.array([[[pt_xy[0], pt_xy[1]]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, matrix)
    return int(out[0][0][0]), int(out[0][0][1])


# -----------------------------
# 2) 입력 소스 확인 (비디오 or 이미지)
# -----------------------------
use_video = False
cap = None

if os.path.exists(VIDEO_PATH):
    cap = cv2.VideoCapture(VIDEO_PATH)
    if cap.isOpened():
        use_video = True
        print(f"비디오 파일 실행: {VIDEO_PATH}")

if not use_video:
    print(f"비디오가 없어 단일 이미지를 처리합니다: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"테스트용 비디오 및 이미지를 찾을 수 없습니다: {IMAGE_PATH}")


# -----------------------------
# 3) 메인 프레임 처리 함수
# -----------------------------
def process_frame(frame):
    frame = cv2.resize(frame, (W, H))

    # (A) 워프 + 역워프 행렬 구하기
    warp_frame, matrix = warp_image(frame, SRC_POINTS, W, H)
    matrix_inv = np.linalg.inv(matrix)

    # (B) 워프된 화면에서 하단 ROI 영역 잘라내기
    h, w = warp_frame.shape[:2]
    x_center = w // 2

    y_start = int(h * ROI_Y_RATIO)
    roi = warp_frame[y_start:h, 0:w]

    # (C) 그레이스케일 변환 + 가우시안 블러
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if BLUR_KSIZE is not None:
        gray = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)

    # (D) 이진화 (Adaptive Thresholding or Fixed Thresholding)
    if USE_ADAPTIVE:
        cur_blockSize = BLOCK_SIZE
        if cur_blockSize % 2 == 0:
            cur_blockSize += 1
        if cur_blockSize < 3:
            cur_blockSize = 3

        mask = cv2.adaptiveThreshold(
            gray, 255,
            ADAPTIVE_METHOD,
            ADAPTIVE_TYPE,
            cur_blockSize,
            C_VALUE
        )
    else:
        _, mask = cv2.threshold(gray, THRESH, 255, cv2.THRESH_BINARY)

    # (E) 모폴로지 연산 (OPEN -> CLOSE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)
    if MORPH_ITER_OPEN > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITER_OPEN)
    if MORPH_ITER_CLOSE > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITER_CLOSE)

    # (F) 컨투어 탐색 및 무게중심 계산
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []          # ROI 좌표계 내 중심점 (cx, cy)
    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        points.append((cx, cy))
        valid_contours.append(cnt)

    # X좌표 기준 오름차순 정렬 (왼쪽 차선 → 오른쪽 차선)
    points.sort(key=lambda p: p[0])

    # (G) 좌/우 차선 및 중앙점 워프 좌표 추출
    left_pt_warp = None
    right_pt_warp = None
    center_pt_warp = None

    if len(points) >= 1:
        left_roi = points[0]
        right_roi = points[-1]

        left_pt_warp = (left_roi[0], left_roi[1] + y_start)
        right_pt_warp = (right_roi[0], right_roi[1] + y_start)

        center_pt_warp = (
            int((left_pt_warp[0] + right_pt_warp[0]) / 2),
            int((left_pt_warp[1] + right_pt_warp[1]) / 2),
        )

    # (H) BEV 시각화 영상 생성
    vis_warp = warp_frame.copy()
    cv2.rectangle(vis_warp, (0, y_start), (w - 1, h - 1), (255, 255, 0), 2)
    cv2.line(vis_warp, (x_center, y_start), (x_center, h), (0, 0, 255), 2)

    # 차선 영역 전체를 흰색으로 채우기 (thickness = -1)
    for cnt in valid_contours:
        cnt_shifted = cnt.copy()
        cnt_shifted[:, 0, 1] += y_start
        cv2.drawContours(vis_warp, [cnt_shifted], -1, (255, 255, 255), -1)

    # 각 컨투어 무게중심 표시
    for (cx, cy) in points:
        cv2.circle(vis_warp, (cx, cy + y_start), 6, (0, 255, 255), -1)

    # 좌/우/중앙점 및 조향 오차 계산 표시
    if left_pt_warp is not None:
        cv2.circle(vis_warp, left_pt_warp, 10, (255, 0, 0), -1)
        cv2.circle(vis_warp, right_pt_warp, 10, (0, 255, 0), -1)
        cv2.circle(vis_warp, center_pt_warp, 12, (0, 255, 255), -1)

        error = x_center - center_pt_warp[0]
        steering = np.clip(error, -200, 200) / 200.0

        cv2.putText(vis_warp, f"error={error}px steering={steering:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(vis_warp, "Lane not detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # (I) 원본 영상에 좌/우/중앙점 역투영 시각화
    vis_orig = frame.copy()

    # 원본 변환 다각형 영역 표시
    cv2.polylines(vis_orig, [SRC_POINTS.astype(np.int32)], True, (255, 255, 0), 2)

    if left_pt_warp is not None:
        left_orig = project_point(left_pt_warp, matrix_inv)
        right_orig = project_point(right_pt_warp, matrix_inv)
        center_orig = project_point(center_pt_warp, matrix_inv)

        cv2.circle(vis_orig, left_orig, 10, (255, 0, 0), -1)
        cv2.circle(vis_orig, right_orig, 10, (0, 255, 0), -1)
        cv2.circle(vis_orig, center_orig, 12, (0, 255, 255), -1)

        cv2.putText(vis_orig, "L", (left_orig[0] + 8, left_orig[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(vis_orig, "R", (right_orig[0] + 8, right_orig[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_orig, "C", (center_orig[0] + 8, center_orig[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 결과 화면 출력
    cv2.imshow("Original + Back-Projected (L/R/C)", vis_orig)
    cv2.imshow("Bird's Eye + Contours (L/R/C)", vis_warp)
    cv2.imshow("Mask (ROI) + Adaptive + Morph", mask)


# -----------------------------
# 4) 실행 루프
# -----------------------------
if use_video:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        process_frame(frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            break
    cap.release()
else:
    frame = cv2.imread(IMAGE_PATH)
    process_frame(frame)
    print("종료하려면 아무 키나 눌러주세요.")
    cv2.waitKey(0)

cv2.destroyAllWindows()
