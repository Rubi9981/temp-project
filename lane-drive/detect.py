"""차선 검출 백엔드.

모두 같은 시그니처를 따른다: fn(mask) -> LaneResult.
입력 mask 는 ROI 크기의 이진 영상이고, 출력 x좌표는 BEV 전체 좌표계와 같다
(ROI는 y만 자르므로 x는 보존된다).

centroid_contours 가 baseline 이다.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

import config as cfg


@dataclass
class LaneResult:
    status: str                 # 'ok' | 'single' | 'fail'
    left_x: float = None        # 왼쪽 차선 위치
    right_x: float = None       # 오른쪽 차선 위치
    center_x: float = None      # 차선 중심 위치
    eval_y: float = None         # 위 x값들을 잰 ROI 내 y
    width: float = None          # right_x - left_x (status='ok'일 때만 의미 있음)
    fit_left: np.ndarray = None     # 왼쪽 차선의 2차 다항식 계수 x=f(y) (sliding_window 전용)
    fit_right: np.ndarray = None    # 오른쪽 차선의 2차 다항식
    fit_center: np.ndarray = None   # 주행에 사용할 중심선. (Pure Pursuit에서 사용) np.polyval(fit_center, y) -> x
    viz: dict = field(default_factory=dict)   # 시각화용 부산물


def count_blobs(mask, min_area=None):
    """마스크에서 유효 면적 이상인 덩어리 수. 이진화 품질의 검출기 무관 지표."""
    if min_area is None:
        min_area = cfg.MIN_AREA
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if cv2.contourArea(c) >= min_area)


# -----------------------------
# baseline
# -----------------------------
def centroid_contours(mask, min_area=None):
    """baseline — project/lane_detector_v2.py 방식.

    ROI 전체 컨투어의 무게중심을 쓰고, x로 정렬해 최좌측/최우측을 차선으로 본다.

    원본의 `len(points) >= 1` 동작을 그대로 재현한다. 덩어리가 하나뿐이면
    left = right = center 가 되어 조향이 크게 튀는데, 이걸 고쳐버리면
    baseline이 실제보다 좋아 보여 비교가 부정직해진다.
    """
    if min_area is None:
        min_area = cfg.MIN_AREA

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    valid = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        points.append((M['m10'] / M['m00'], M['m01'] / M['m00']))
        valid.append(cnt)

    points.sort(key=lambda p: p[0])

    if not points:
        return LaneResult(status='fail', viz={'contours': valid, 'points': points})

    left, right = points[0], points[-1]
    center_x = (left[0] + right[0]) / 2
    eval_y = (left[1] + right[1]) / 2
    status = 'single' if len(points) == 1 else 'ok'

    return LaneResult(
        status=status,
        left_x=left[0],
        right_x=right[0],       # status='single'이면 left와 같은 점이다 (원본 동작)
        center_x=center_x,
        eval_y=eval_y,
        width=(right[0] - left[0]) if status == 'ok' else None,
        viz={'contours': valid, 'points': points},
    )


# -----------------------------
# 신규
# -----------------------------
def sliding_window(mask,
                   nwindows=None, margin=None, minpix=None,
                   min_peak=None, min_fitpix=None,
                   lane_width_px=None):
    """하단 히스토그램으로 시작점을 잡고 위로 윈도우를 옮겨가며 차선을 추적한다.

    윈도우 밖의 밝은 덩어리(도로 한가운데 반사광)는 자연히 배제된다.
    부산물로 2차 다항식 계수가 나오므로 곡률을 쓸 수 있다.

    한쪽만 잡히면 알려진 차선 폭으로 반대쪽을 외삽한다 — baseline이
    좌우를 붕괴시켜 조향을 풀락으로 보내던 지점이다.
    """
    nwindows = nwindows or cfg.SW_NWINDOWS
    margin = margin or cfg.SW_MARGIN
    minpix = minpix or cfg.SW_MINPIX
    min_peak = min_peak or cfg.SW_MIN_PEAK
    min_fitpix = min_fitpix or cfg.SW_MIN_FITPIX
    lane_width_px = lane_width_px or cfg.LANE_WIDTH_PX

    h, w = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8)

    # 하단 절반의 열별 픽셀 수 -> 좌/우 시작점
    histogram = binary[h // 2:, :].sum(axis=0)
    midpoint = w // 2

    bases = {}
    for side, sl, offset in (('left', slice(0, midpoint), 0),
                             ('right', slice(midpoint, w), midpoint)):
        seg = histogram[sl]
        if seg.size and seg.max() >= min_peak:
            bases[side] = int(np.argmax(seg)) + offset

    # nonzero() 는 행 우선이라 nonzero_y 가 이미 오름차순이다. 덕분에 윈도우의
    # y 범위를 searchsorted 로 잘라낼 수 있고, x 비교는 그 밴드(전체의 1/nwindows)
    # 에서만 하면 된다. 예전에는 윈도우 18개(좌우x9)마다 전체 nonzero 배열을
    # 네 번씩 비교했다 — 마스크가 조밀하면 이게 검출 시간의 대부분이었다.
    nonzero_y, nonzero_x = binary.nonzero()
    window_h = h // nwindows

    lane_px = {}
    boxes = []

    for side, base in bases.items():
        current = base
        collected = []
        for i in range(nwindows):
            y_low = h - (i + 1) * window_h
            y_high = h - i * window_h
            x_low = current - margin
            x_high = current + margin
            boxes.append((x_low, y_low, x_high, y_high))

            lo = np.searchsorted(nonzero_y, y_low, 'left')
            hi = np.searchsorted(nonzero_y, y_high, 'left')
            band_x = nonzero_x[lo:hi]
            sel = ((band_x >= x_low) & (band_x < x_high)).nonzero()[0]
            collected.append(sel + lo)          # 원래 배열 기준 인덱스로 되돌린다

            if len(sel) > minpix:
                current = int(band_x[sel].mean())

        idx = np.concatenate(collected) if collected else np.array([], int)
        if len(idx) >= min_fitpix:
            lane_px[side] = idx

    # 2차 다항식 피팅 (x = f(y))
    fits = {}
    for side, idx in lane_px.items():
        ys, xs = nonzero_y[idx], nonzero_x[idx]
        if len(np.unique(ys)) < 3:      # 서로 다른 y가 3개 미만이면 2차 피팅 불가
            continue
        fits[side] = np.polyfit(ys, xs, 2)

    eval_y = h / 2.0
    viz = {'boxes': boxes, 'lane_px': lane_px, 'nonzero': (nonzero_y, nonzero_x)}

    left_x = np.polyval(fits['left'], eval_y) if 'left' in fits else None
    right_x = np.polyval(fits['right'], eval_y) if 'right' in fits else None

    if left_x is None and right_x is None:
        return LaneResult(status='fail', eval_y=eval_y, viz=viz)

    if left_x is not None and right_x is not None:
        # np.polyval 은 계수에 선형이므로, 계수를 평균내면 두 곡선의 중간 곡선이 된다
        fit_center = (fits['left'] + fits['right']) / 2
        return LaneResult(
            status='ok',
            left_x=float(left_x), right_x=float(right_x),
            center_x=float((left_x + right_x) / 2),
            eval_y=eval_y,
            width=float(right_x - left_x),
            fit_left=fits.get('left'), fit_right=fits.get('right'),
            fit_center=fit_center,
            viz=viz,
        )

    # 한쪽만 잡힌 경우 — 알려진 차선 폭으로 반대쪽을 세운다.
    # 상수항만 밀어 평행 이동시키는 근사다. BEV에서 차선이 수직에 가까울 때만
    # 정확하며, 곡률이 큰 구간에서는 법선 방향 간격이 이보다 좁아진다.
    half = lane_width_px / 2
    if left_x is not None:
        right_x = left_x + lane_width_px
        fit_center = fits['left'].copy()
        fit_center[-1] += half
    else:
        left_x = right_x - lane_width_px
        fit_center = fits['right'].copy()
        fit_center[-1] -= half

    return LaneResult(
        status='single',
        left_x=float(left_x), right_x=float(right_x),
        center_x=float((left_x + right_x) / 2),
        eval_y=eval_y,
        width=None,                     # 외삽한 폭이라 지표에 넣지 않는다
        fit_left=fits.get('left'), fit_right=fits.get('right'),
        fit_center=fit_center,
        viz=viz,
    )


DETECTORS = {
    'centroid': centroid_contours,
    'sliding': sliding_window,
}

BASELINE = 'centroid'


# -----------------------------
# 조향
# -----------------------------
def lane_error(center_x, width=cfg.W):
    """기존 식 유지: error = 화면중심 - 차선중심, steering = clip/200."""
    if center_x is None:
        return None, None
    error = width // 2 - center_x
    steering = float(np.clip(error, -cfg.STEER_CLIP, cfg.STEER_CLIP)) / cfg.STEER_CLIP
    return error, steering
