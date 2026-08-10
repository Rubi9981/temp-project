"""BEV 변환과 BEV 기하 품질 지표.

warp_image / project_point 은 project/lane_detector_v2.py 에서 그대로 이식했다.
품질 지표는 "직선 구간에서 차선 폭은 y에 무관하게 일정해야 한다"는
물리적 제약을 이용한다.
"""
import cv2
import numpy as np

import config as cfg

# 폭 지표를 측정할 BEV 행. ROI(y >= 264) 안쪽만 사용한다.
METRIC_ROWS = tuple(range(270, 470, 20))


# -----------------------------
# 1) 변환
# -----------------------------
def warp_image(image, src_pts=None, width=cfg.W, height=cfg.H):
    """원본 이미지를 BEV로 변환하고 변환 행렬을 함께 돌려준다."""
    if src_pts is None:
        src_pts = cfg.get_src_points()

    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(np.asarray(src_pts, np.float32), dst_points)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, matrix


def project_point(pt_xy, matrix):
    """점 하나를 주어진 행렬로 변환한다 (BEV -> 원본 역투영 등)."""
    pt = np.array([[[pt_xy[0], pt_xy[1]]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, matrix)
    return int(out[0][0][0]), int(out[0][0][1])


def project_points(pts_xy, matrix):
    """점 여러 개를 한 번에 변환한다. pts_xy: (N,2) -> (N,2) int32."""
    pts = np.asarray(pts_xy, np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, matrix)
    return out.reshape(-1, 2).astype(np.int32)


def curve_points(fit, y_from, y_to, x_offset=0, y_offset=0, step=4):
    """x=f(y) 다항식을 (N,2) 점열로 편다. y_offset 은 ROI -> BEV 보정용."""
    ys = np.arange(y_from, y_to, step, dtype=float)
    xs = np.polyval(fit, ys) + x_offset
    return np.stack([xs, ys + y_offset], axis=1)


# -----------------------------
# 2b) BEV 픽셀 <-> 차량 좌표 (cm)
# -----------------------------
# 차량 좌표계: 원점 = 뒤축 중심, X = 전방(+), Y = 좌측(+).
# BEV 는 y 가 아래로 갈수록 차에 가까우므로 X 는 (H-1 - y) 에 비례한다.
# x 와 y 의 cm 환산 계수가 다른 이유는, dst 사각형을 640x480 전체로 잡아
# BEV 를 만들었기 때문이다 (실세계 영역의 가로:세로 비와 무관하게 4:3 으로 강제).

def bev_to_vehicle(x_px, y_px, metric=None):
    """BEV 픽셀 -> (X 전방 cm, Y 좌측 cm). 스칼라와 배열 모두 받는다."""
    m = metric if metric is not None else cfg.get_metric()
    X = m.rear_axle_offset_cm + (cfg.H - 1 - np.asarray(y_px, float)) / m.px_per_cm_y
    Y = (m.vehicle_center_x_px - np.asarray(x_px, float)) / m.px_per_cm_x
    return X, Y


def vehicle_to_bev(X_cm, Y_cm, metric=None):
    """(X 전방 cm, Y 좌측 cm) -> BEV 픽셀. 원호를 그리는 데 쓴다."""
    m = metric if metric is not None else cfg.get_metric()
    x_px = m.vehicle_center_x_px - np.asarray(Y_cm, float) * m.px_per_cm_x
    y_px = (cfg.H - 1) - (np.asarray(X_cm, float) - m.rear_axle_offset_cm) * m.px_per_cm_y
    return x_px, y_px


def load_bev(path, src_pts=None, already_bev=False):
    """이미지를 읽어 BEV를 돌려준다.

    already_bev=True 면 warp를 건너뛴다 (captures_bev/ 의 기존 산출물용).
    반환: (bev, matrix, original) — already_bev일 때 matrix는 None.
    """
    img = cv2.imread(path)
    if img is None:
        return None, None, None

    img = cv2.resize(img, (cfg.W, cfg.H))
    if already_bev:
        return img, None, img

    bev, matrix = warp_image(img, src_pts)
    return bev, matrix, img


def roi_of(bev):
    """BEV 하단 ROI를 잘라 (roi, y_start) 로 돌려준다."""
    h = bev.shape[0]
    y_start = int(h * cfg.ROI_Y_RATIO)
    return bev[y_start:h, :], y_start


# -----------------------------
# 2) 행 단위 차선 위치 추출
# -----------------------------
def row_lane_centers(mask_row, min_run=4, gap=10):
    """이진 마스크 한 행에서 흰 픽셀 덩어리들의 중심 x 목록을 돌려준다."""
    xs = np.where(mask_row > 0)[0]
    if len(xs) == 0:
        return []
    groups = np.split(xs, np.where(np.diff(xs) > gap)[0] + 1)
    return [float(g.mean()) for g in groups if len(g) >= min_run]


def lane_width_profile(bev_bgr, binarize_fn, rows=METRIC_ROWS):
    """여러 행에서 (좌끝, 우끝) 을 재어 폭과 중심의 프로파일을 만든다.

    반환: (ys, widths, centers) — 각각 np.ndarray. 측정 실패한 행은 빠진다.
    """
    mask = binarize_fn(bev_bgr)
    ys, widths, centers = [], [], []
    for y in rows:
        if y >= mask.shape[0]:
            continue
        c = row_lane_centers(mask[y])
        if len(c) < 2:
            continue
        ys.append(y)
        widths.append(c[-1] - c[0])
        centers.append((c[-1] + c[0]) / 2.0)
    return np.array(ys, float), np.array(widths, float), np.array(centers, float)


def _slope(ys, values, min_points=4):
    if len(ys) < min_points:
        return None
    return float(np.polyfit(ys, values, 1)[0])


def width_slope(ys, widths):
    """폭의 y방향 기울기. 직선 구간에서 0이어야 한다.

    양수 = 아래(근거리)로 갈수록 차선이 벌어짐 = BEV가 원근을 덜 편 상태.
    """
    return _slope(ys, widths)


def center_slope(ys, centers):
    """차선 중심의 y방향 기울기. 직선 구간 판별에 쓴다 (0에 가까울수록 직선)."""
    return _slope(ys, centers)


# -----------------------------
# 3) 사다리꼴 파라미터화 (tune_src.py 용)
# -----------------------------
def trapezoid_to_src(cx, top_y, top_hw, bot_y, bot_hw):
    """해석 가능한 5개 값으로 SRC_POINTS를 만든다.

    8자유도를 그대로 최적화하면 프록시 지표에 과적합되므로,
    좌우 대칭 사다리꼴로 제한해 5자유도로 줄인다.
    """
    return np.array([
        [cx - top_hw, top_y],   # 좌상
        [cx + top_hw, top_y],   # 우상
        [cx + bot_hw, bot_y],   # 우하
        [cx - bot_hw, bot_y],   # 좌하
    ], dtype=np.float32)


def src_to_trapezoid(src):
    """SRC_POINTS를 5개 파라미터로 근사한다 (최적화 초기값용).

    현재 기본 SRC_POINTS는 완전 대칭이 아니므로 평균으로 근사하며,
    이 과정에서 몇 px의 비대칭 정보가 버려진다.
    """
    src = np.asarray(src, float)
    tl, tr, br, bl = src
    top_y = (tl[1] + tr[1]) / 2
    bot_y = (br[1] + bl[1]) / 2
    top_hw = (tr[0] - tl[0]) / 2
    bot_hw = (br[0] - bl[0]) / 2
    cx = (tl[0] + tr[0] + bl[0] + br[0]) / 4
    return cx, top_y, top_hw, bot_y, bot_hw
