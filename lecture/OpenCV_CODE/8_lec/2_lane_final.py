import cv2
import numpy as np

def warp_image(image, src_points, width=640, height=480):
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, H, (width, height))
    return warped, H

def project_point(pt_xy, H):
    pt = np.array([[[pt_xy[0], pt_xy[1]]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return int(out[0][0][0]), int(out[0][0][1])

# -----------------------------
# Warp 기준 좌표
# -----------------------------
src_points = np.array([
    [200, 240],
    [420, 240],
    [639, 420],
    [0,   420]
], dtype=np.float32)

# -----------------------------
# 파라미터
# -----------------------------
roi_y_ratio = 0.55
min_area = 200  # 멀티행이면 보통 조금 낮추는 게 좋음

blur_ksize = (45, 45)
morph_ksize = (5, 5)
morph_iter_open = 1
morph_iter_close = 2

use_adaptive = True
thresh = 127
adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
adaptive_type = cv2.THRESH_BINARY
blockSize = 31
C = -2

# 멀티행 설정
num_rows = 8

# 한쪽만 잡힐 때 처리 모드
mode = "predict"   # "strict" 또는 "predict"

# 예측모드에서 사용할 폭 추정(기본값/보호장치)
default_lane_width = 220   # 픽셀 (워프 화면 기준 대략값)
min_lane_width = 80
max_lane_width = 400

cap = cv2.VideoCapture('source/lane_test.mp4')
if not cap.isOpened():
    raise FileNotFoundError("비디오 파일을 열 수 없습니다: source/lane_test.mp4")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame = cv2.resize(frame, (640, 480))

    warp_frame, H = warp_image(frame, src_points, 640, 480)
    H_inv = np.linalg.inv(H)

    h, w = warp_frame.shape[:2]
    x_center = w // 2

    y_start = int(h * roi_y_ratio)
    roi = warp_frame[y_start:h, 0:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, blur_ksize, 0)

    if use_adaptive:
        if blockSize % 2 == 0:
            blockSize += 1
        if blockSize < 3:
            blockSize = 3
        mask = cv2.adaptiveThreshold(gray, 255, adaptive_method, adaptive_type, blockSize, C)
    else:
        _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_ksize)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iter_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iter_close)

    # -------------------------------------------------------
    # ★ 멀티행: 아래쪽 행부터 위로 처리하며 lane_width 추정값 누적
    # -------------------------------------------------------
    roi_h, roi_w = mask.shape[:2]
    row_h = max(1, roi_h // num_rows)

    lefts_warp = []
    rights_warp = []
    centers_warp = []

    # "아래쪽(차에 가까운)"에서 먼저 얻은 차선 폭으로 위쪽을 예측
    estimated_lane_width = None

    # 아래 -> 위로 진행 (중요!)
    for i in range(num_rows - 1, -1, -1):
        y0 = i * row_h
        y1 = roi_h if i == num_rows - 1 else (i + 1) * row_h

        strip = mask[y0:y1, 0:roi_w]
        contours, _ = cv2.findContours(strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # strip 내에서 "유효 컨투어 중심점들" 수집
        pts = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            pts.append((cx, cy))

        if len(pts) == 0:
            continue

        # x로 정렬
        pts.sort(key=lambda p: p[0])

        # ---- left/right 후보 만들기 ----
        # pts가 많으면 양 끝이 노이즈일 수 있으니,
        # "카메라 중심 기준 왼쪽에서 가장 가까운 점" / "오른쪽에서 가장 가까운 점"을 쓰면 더 안정적임
        left_candidates = [p for p in pts if p[0] < x_center]
        right_candidates = [p for p in pts if p[0] >= x_center]

        left_strip = max(left_candidates, key=lambda p: p[0]) if len(left_candidates) > 0 else None
        right_strip = min(right_candidates, key=lambda p: p[0]) if len(right_candidates) > 0 else None

        # strip -> warp 전체 좌표로 변환 함수
        def to_warp(p):
            return (p[0], y_start + y0 + p[1])

        left_warp = to_warp(left_strip) if left_strip is not None else None
        right_warp = to_warp(right_strip) if right_strip is not None else None

        # ---- lane_width 업데이트 (둘 다 있을 때만) ----
        if left_warp is not None and right_warp is not None:
            lane_w = right_warp[0] - left_warp[0]
            lane_w = int(np.clip(lane_w, min_lane_width, max_lane_width))
            estimated_lane_width = lane_w

            center_warp = (int((left_warp[0] + right_warp[0]) / 2),
                           int((left_warp[1] + right_warp[1]) / 2))

            lefts_warp.append(left_warp)
            rights_warp.append(right_warp)
            centers_warp.append(center_warp)
            continue

        # ---- 한쪽만 있을 때 처리 ----
        if mode == "strict":
            # 한쪽만 있으면 이 행은 버림
            continue

        # mode == "predict"
        lane_w = estimated_lane_width if estimated_lane_width is not None else default_lane_width

        if left_warp is not None and right_warp is None:
            # 오른쪽을 추정
            right_warp = (int(left_warp[0] + lane_w), left_warp[1])
            center_warp = (int((left_warp[0] + right_warp[0]) / 2), left_warp[1])

            lefts_warp.append(left_warp)
            rights_warp.append(right_warp)
            centers_warp.append(center_warp)

        elif right_warp is not None and left_warp is None:
            # 왼쪽을 추정
            left_warp = (int(right_warp[0] - lane_w), right_warp[1])
            center_warp = (int((left_warp[0] + right_warp[0]) / 2), right_warp[1])

            lefts_warp.append(left_warp)
            rights_warp.append(right_warp)
            centers_warp.append(center_warp)

    # 우리가 아래->위로 돌았으니, 시각화/제어용으로 y 오름차순(위->아래)로 정렬
    centers_warp.sort(key=lambda p: p[1])
    lefts_warp.sort(key=lambda p: p[1])
    rights_warp.sort(key=lambda p: p[1])

    # -----------------------------
    # 시각화 (워프)
    # -----------------------------
    vis_warp = warp_frame.copy()
    cv2.rectangle(vis_warp, (0, y_start), (w - 1, h - 1), (255, 255, 0), 2)
    cv2.line(vis_warp, (x_center, y_start), (x_center, h), (0, 0, 255), 2)

    for (lx, ly) in lefts_warp:
        cv2.circle(vis_warp, (lx, ly), 7, (255, 0, 0), -1)
    for (rx, ry) in rights_warp:
        cv2.circle(vis_warp, (rx, ry), 7, (0, 255, 0), -1)
    for (cx, cy) in centers_warp:
        cv2.circle(vis_warp, (cx, cy), 9, (0, 255, 255), -1)

    # 제어는 가장 아래(center y가 가장 큰 것)
    steering = None
    if len(centers_warp) > 0:
        center_for_control = max(centers_warp, key=lambda p: p[1])
        error = x_center - center_for_control[0]
        steering = np.clip(error, -200, 200) / 200.0
        cv2.circle(vis_warp, center_for_control, 12, (0, 165, 255), -1)
        cv2.putText(vis_warp, f"mode={mode} error={error}px steering={steering:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(vis_warp, f"mode={mode} Lane not detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # -----------------------------
    # 원본에 역투영 점 표시
    # -----------------------------
    vis_orig = frame.copy()
    for p in src_points.astype(int):
        cv2.circle(vis_orig, tuple(p), 6, (255, 255, 0), -1)

    for c_warp in centers_warp:
        c_orig = project_point(c_warp, H_inv)
        cv2.circle(vis_orig, c_orig, 7, (0, 255, 255), -1)

    for l_warp in lefts_warp:
        l_orig = project_point(l_warp, H_inv)
        cv2.circle(vis_orig, l_orig, 5, (255, 0, 0), -1)

    for r_warp in rights_warp:
        r_orig = project_point(r_warp, H_inv)
        cv2.circle(vis_orig, r_orig, 5, (0, 255, 0), -1)

    # 출력
    cv2.imshow("Original + Back-Projected (Row, strict/predict)", vis_orig)
    cv2.imshow("Bird's Eye (Row, strict/predict)", vis_warp)
    cv2.imshow("Mask (ROI)", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
