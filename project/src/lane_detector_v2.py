import cv2
import numpy as np
import os

# ==============================================================================
# [설정 및 파라미터 튜닝] - lane_detector_v2 (수동 키 조작 이미지 검증)
# ==============================================================================

# 1. 경로 설정
BEV_DIR = 'captures_bev'                            # 순회 탐색할 BEV 이미지 폴더
IMAGE_PATH = 'captures/frame_20260803_162053.jpg'  # 단일 파일 fallback

# 2. 이미지 규격 및 BEV (Warp) 기준 좌표 ([좌상, 우상, 우하, 좌하] 순서)
W, H = 640, 480
SRC_POINTS = np.array([
    [36, 262],   # 좌상단
    [588, 269],  # 우상단
    [605, 406],  # 우하단
    [17, 413]    # 좌하단
], dtype=np.float32)

# 3. 측정된 차선 HSV 마스킹 범위 설정
LOWER_HSV = np.array([50, 0, 135], dtype=np.uint8)
UPPER_HSV = np.array([105, 50, 255], dtype=np.uint8)

# 4. 차선 검출 영역 및 면적 파라미터
ROI_Y_RATIO = 0.55     # BEV 이미지 하단 ROI 시작 비율 (0.55 = 하단 45% 영역 사용)
MIN_AREA = 300          # 노이즈 제거용 최소 차선 컨투어 면적

# 5. 블러 / 모폴로지 커널 설정
BLUR_KSIZE = (5, 5)
MORPH_KSIZE = (5, 5)
MORPH_ITER_OPEN = 1
MORPH_ITER_CLOSE = 2

# 6. 기본 작동 모드 (False: 키 누를 때만 넘어가기, True: 자동 재생)
AUTO_PLAY = False
PLAY_DELAY_MS = 200     # AUTO_PLAY = True 일 때의 자동 재생 간격 (ms)

# ==============================================================================


# -----------------------------
# 1) BEV 변환 및 역투영 함수
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
# 2) 프레임 검출 및 시각화 함수
# -----------------------------
def process_bev_frame(frame, filename_info="", is_bev_already=True):
    frame = cv2.resize(frame, (W, H))

    if is_bev_already:
        warp_frame = frame.copy()
        dst_points = np.array([[0,0], [W-1,0], [W-1,H-1], [0,H-1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(SRC_POINTS, dst_points)
        matrix_inv = np.linalg.inv(matrix)
    else:
        warp_frame, matrix = warp_image(frame, SRC_POINTS, W, H)
        matrix_inv = np.linalg.inv(matrix)

    # (B) 하단 ROI 잘라내기
    h, w = warp_frame.shape[:2]
    x_center = w // 2

    y_start = int(h * ROI_Y_RATIO)
    roi = warp_frame[y_start:h, 0:w]

    # (C) 가우시안 블러
    if BLUR_KSIZE is not None:
        blurred_roi = cv2.GaussianBlur(roi, BLUR_KSIZE, 0)
    else:
        blurred_roi = roi.copy()

    # (D) BGR -> HSV 변환 및 마스킹
    hsv_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, LOWER_HSV, UPPER_HSV)

    # (E) 모폴로지 연산
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)
    if MORPH_ITER_OPEN > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITER_OPEN)
    if MORPH_ITER_CLOSE > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITER_CLOSE)

    # (F) 컨투어 탐색 및 무게중심 계산
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []          # ROI 좌표계 내 중심점
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

    # (H) BEV 시각화 생성
    vis_warp = warp_frame.copy()
    cv2.rectangle(vis_warp, (0, y_start), (w - 1, h - 1), (255, 255, 0), 2)
    cv2.line(vis_warp, (x_center, y_start), (x_center, h), (0, 0, 255), 2)

    # 차선 영역 전체를 흰색으로 채우기
    for cnt in valid_contours:
        cnt_shifted = cnt.copy()
        cnt_shifted[:, 0, 1] += y_start
        cv2.drawContours(vis_warp, [cnt_shifted], -1, (255, 255, 255), -1)

    # 중심점 표시
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

    # 파일 정보 상단 표시
    if filename_info:
        cv2.putText(vis_warp, f"File: {filename_info}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 결과 화면 출력
    cv2.imshow("Bird's Eye + Detection (L/R/C)", vis_warp)
    cv2.imshow("Mask (ROI) + HSV inRange", mask)


# -----------------------------
# 3) 수동 조작 탐색 메인 함수
# -----------------------------
def main():
    global AUTO_PLAY

    if os.path.exists(BEV_DIR):
        file_list = sorted([
            f for f in os.listdir(BEV_DIR)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
    else:
        file_list = []

    if not file_list:
        print(f"'{BEV_DIR}' 폴더에 이미지 파일이 없어 단일 이미지('{IMAGE_PATH}')를 표시합니다.")
        frame = cv2.imread(IMAGE_PATH)
        if frame is None:
            raise FileNotFoundError("이미지를 찾을 수 없습니다.")
        process_bev_frame(frame, filename_info=IMAGE_PATH, is_bev_already=False)
        print("종료하려면 아무 키나 눌러주세요.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    print("=" * 65)
    print(f"'{BEV_DIR}' 폴더 총 {len(file_list)}개 이미지 (수동 넘김 모드)")
    print("  - [d] / [Spacebar] / [오른쪽 화살표] : 다음 이미지 ▶")
    print("  - [a] / [왼쪽 화살표]                : 이전 이미지 ◀")
    print("  - [s]                              : 자동 재생 / 수동 모드 토글")
    print("  - [q] / [ESC]                      : 종료")
    print("=" * 65)

    idx = 0

    while True:
        filename = file_list[idx]
        file_path = os.path.join(BEV_DIR, filename)

        frame = cv2.imread(file_path)
        if frame is not None:
            mode_str = "자동재생" if AUTO_PLAY else "수동모드"
            info_text = f"[{idx+1}/{len(file_list)}] {filename} ({mode_str})"
            process_bev_frame(frame, filename_info=info_text, is_bev_already=True)

        # AUTO_PLAY 모드에 따라 키 대기 시간 설정 (수동 모드는 0 = 키 누를 때까지 대기)
        wait_time = PLAY_DELAY_MS if AUTO_PLAY else 0
        key = cv2.waitKey(wait_time) & 0xFF

        # 키 조작 처리
        if key == 27 or key == ord('q'):    # q 또는 ESC (종료)
            break
        elif key == ord('s'):               # s 키 (자동 재생 / 수동 전환)
            AUTO_PLAY = not AUTO_PLAY
            print(f"모드 변경: {'자동 재생 모드' if AUTO_PLAY else '수동 넘김 모드'}")
        elif key in [ord('d'), ord('n'), 32, 13, 83]:  # d, n, Space, Enter, 오른쪽 화살표 -> 다음
            idx = (idx + 1) % len(file_list)
        elif key in [ord('a'), ord('p'), 81]:           # a, p, 왼쪽 화살표 -> 이전
            idx = (idx - 1 + len(file_list)) % len(file_list)
        else:
            # 자동 재생 모드일 때만 키 입력 없이 다음으로 이동
            if AUTO_PLAY:
                idx = (idx + 1) % len(file_list)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
