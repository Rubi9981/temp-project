import cv2
import numpy as np

# -----------------------------
# 1) Bird's Eye View (Warp) 함수
# -----------------------------
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
    """
    pt_xy: (x, y)
    H: 3x3 perspective matrix
    return: (x, y) int
    """
    pt = np.array([[[pt_xy[0], pt_xy[1]]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return int(out[0][0][0]), int(out[0][0][1])


# -----------------------------
# 2) Warp 기준 좌표
# -----------------------------
src_points = np.array([
    [210, 220],  # 좌상단: 왼쪽 차선 시작점
    [410, 220],  # 우상단: 오른쪽 차선 시작점
    [639, 400],  # 우하단: 오른쪽 바닥 가까운 지점
    [0, 400] 
], dtype=np.float32)

# -----------------------------
# 3) 컨투어 파라미터
# -----------------------------
roi_y_ratio = 0.55
thresh = 127
min_area = 300

cap = cv2.VideoCapture('source/lane_test.mp4')
if not cap.isOpened():
    raise FileNotFoundError("비디오 파일을 열 수 없습니다: source/lane_test.mp4")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame = cv2.resize(frame, (640, 480))

    # (A) 워프 + 역워프 행렬
    warp_frame, H = warp_image(frame, src_points, 640, 480)
    H_inv = np.linalg.inv(H)

    # (B) 워프된 화면에서 ROI
    h, w = warp_frame.shape[:2]
    x_center = w // 2

    y_start = int(h * roi_y_ratio)
    roi = warp_frame[y_start:h, 0:w]

    # (C) 이진화
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    # (D) 컨투어 / 무게중심
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []          # (cx, cy) in ROI coords
    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        points.append((cx, cy))
        valid_contours.append(cnt)

    # 왼쪽/오른쪽 판단을 위해 x로 정렬
    points.sort(key=lambda p: p[0])

    # (E) left/right/center (warp 좌표계에서)
    left_pt_warp = None
    right_pt_warp = None
    center_pt_warp = None

    if len(points) >= 1:
        # ROI 좌표 -> warp 전체 좌표로 변환 (y에 y_start 더함)
        left_roi = points[0]
        right_roi = points[-1]

        left_pt_warp = (left_roi[0], left_roi[1] + y_start)
        right_pt_warp = (right_roi[0], right_roi[1] + y_start)

        # 가운데 점 (x는 평균, y는 두 점의 평균으로 잡음)
        center_pt_warp = (
            int((left_pt_warp[0] + right_pt_warp[0]) / 2),
            int((left_pt_warp[1] + right_pt_warp[1]) / 2),
        )

    # (F) 워프 화면 시각화
    vis_warp = warp_frame.copy()
    cv2.rectangle(vis_warp, (0, y_start), (w - 1, h - 1), (255, 255, 0), 2)
    cv2.line(vis_warp, (x_center, y_start), (x_center, h), (0, 0, 255), 2)

    # 컨투어 그리기(ROI -> warp 전체로 shift)
    for cnt in valid_contours:
        cnt_shifted = cnt.copy()
        cnt_shifted[:, 0, 1] += y_start
        cv2.drawContours(vis_warp, [cnt_shifted], -1, (0, 255, 0), 2)

    # 모든 중심점 표시(워프)
    for (cx, cy) in points:
        cv2.circle(vis_warp, (cx, cy + y_start), 6, (0, 255, 255), -1)

    # left/right/center 점 표시(워프)
    if left_pt_warp is not None:
        cv2.circle(vis_warp, left_pt_warp, 10, (255, 0, 0), -1)    # Left: 파랑
        cv2.circle(vis_warp, right_pt_warp, 10, (0, 255, 0), -1)   # Right: 초록
        cv2.circle(vis_warp, center_pt_warp, 12, (0, 255, 255), -1)# Center: 노랑

        # steering 계산은 center 기준으로
        error = x_center - center_pt_warp[0]
        steering = np.clip(error, -200, 200) / 200.0

        cv2.putText(vis_warp, f"error={error}px steering={steering:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # (G) 원본 화면에 역투영해서 점 찍기
    vis_orig = frame.copy()

    # 원본에도 src_points 표시(선택: 워프 영역 확인용)
    for p in src_points.astype(int):
        cv2.circle(vis_orig, tuple(p), 6, (255, 255, 0), -1)

    if left_pt_warp is not None:
        left_orig = project_point(left_pt_warp, H_inv)
        right_orig = project_point(right_pt_warp, H_inv)
        center_orig = project_point(center_pt_warp, H_inv)

        # 원본에 3점 찍기
        cv2.circle(vis_orig, left_orig, 10, (255, 0, 0), -1)       # Left: 파랑
        cv2.circle(vis_orig, right_orig, 10, (0, 255, 0), -1)      # Right: 초록
        cv2.circle(vis_orig, center_orig, 12, (0, 255, 255), -1)   # Center: 노랑

        cv2.putText(vis_orig, "L", (left_orig[0] + 8, left_orig[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(vis_orig, "R", (right_orig[0] + 8, right_orig[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_orig, "C", (center_orig[0] + 8, center_orig[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 출력
    cv2.imshow("Original + Back-Projected (L/R/C)", vis_orig)
    cv2.imshow("Bird's Eye + Contours (L/R/C)", vis_warp)
    cv2.imshow("Mask (ROI on Bird's Eye)", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
