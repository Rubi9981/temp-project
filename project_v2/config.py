"""모든 파라미터의 단일 소스.

기존 project/ 에서는 SRC_POINTS가 6개 파일에 복붙되어 있어 재튜닝 시
일부만 수정되면 조용히 어긋났다. 여기서는 이 파일 하나만 본다.

튜닝 결과는 calib.json 에 저장되며, 존재하면 기본값보다 우선한다.
"""
import json
import os

import numpy as np

# -----------------------------
# 경로
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'project')

CAPTURES_DIR = os.path.join(PROJECT_DIR, 'captures')          # 원본 (74장)
CAPTURES_BEV_DIR = os.path.join(PROJECT_DIR, 'captures_bev')  # 기존 BEV 산출물 (68장, stale)

CALIB_PATH = os.path.join(BASE_DIR, 'calib.json')
GT_PATH = os.path.join(BASE_DIR, 'gt.json')

# -----------------------------
# 이미지 규격
# -----------------------------
W, H = 640, 480

# BEV 기준 좌표 [좌상, 우상, 우하, 좌하]
# project/coordinate.py 로 뽑은 원래 값. calib.json 이 있으면 그쪽이 우선한다.
SRC_POINTS_DEFAULT = np.array([
    [36, 262],   # 좌상단
    [588, 269],  # 우상단
    [605, 406],  # 우하단
    [17, 413],   # 좌하단
], dtype=np.float32)

# -----------------------------
# 검출 영역
# -----------------------------
ROI_Y_RATIO = 0.55      # BEV 하단 ROI 시작 비율
MIN_AREA = 300          # 컨투어 최소 면적

BLUR_KSIZE = (5, 5)
MORPH_KSIZE = (5, 5)
MORPH_ITER_OPEN = 1
MORPH_ITER_CLOSE = 2

# -----------------------------
# 이진화 백엔드별 파라미터
# -----------------------------
# hsv_inrange (baseline)
LOWER_HSV = np.array([50, 0, 135], dtype=np.uint8)
UPPER_HSV = np.array([105, 50, 255], dtype=np.uint8)

# adaptive_gray (project/lane_detector.py 방식)
ADAPT_BLOCK = 21        # 홀수
ADAPT_C = -4

# tophat_otsu (신규)
# 커널은 BEV상 차선 폭(약 30~40px)보다 커야 한다. 커널보다 큰 밝은 영역
# (= 천장 조명 반사 얼룩)이 제거된다.
TOPHAT_KSIZE = (51, 51)
# 차선이 아예 없는 프레임에서 Otsu가 노이즈를 이진화하는 것을 막는 하한.
# Otsu가 고른 임계값이 이보다 낮으면 "볼 게 없다"로 판정한다.
TOPHAT_MIN_OTSU = 20

# -----------------------------
# 슬라이딩 윈도우 검출
# -----------------------------
SW_NWINDOWS = 9
SW_MARGIN = 60          # 윈도우 좌우 폭 (px)
SW_MINPIX = 50          # 윈도우 재중심화 최소 픽셀 수
# 히스토그램 피크로 인정할 최소값. 히스토그램은 ROI 하단 절반의 열별 픽셀
# 수이므로 최댓값이 곧 그 절반의 행 수(약 108)다.
SW_MIN_PEAK = 15
SW_MIN_FITPIX = 200     # 다항식 피팅에 필요한 최소 픽셀 수

# BEV상 차선 폭(px). 한쪽 차선만 잡혔을 때 반대쪽 외삽에 쓴다.
# gt.json 의 좌우 라벨 51쌍 실측: mean=457.5 std=25.8 (범위 412~532).
LANE_WIDTH_PX = 457.5

# -----------------------------
# 조향
# -----------------------------
STEER_CLIP = 200        # error를 이 값으로 clip 후 정규화

# -----------------------------
# 정답 라벨 (label_gt.py)
# -----------------------------
# 라벨을 찍을 BEV 행. ROI 시작(264) + ROI 높이의 절반(108) = 372 으로,
# sliding_window 가 차선 위치를 평가하는 행과 일치시킨 값이다.
GT_ROW_BEV = 372

# -----------------------------
# 향후 확장 자리 (체커보드 캘리브레이션 도입 시 채운다)
# -----------------------------
CAMERA_MATRIX = None    # np.ndarray (3,3)
DIST_COEFFS = None      # np.ndarray (5,)


def get_src_points():
    """calib.json 이 있으면 그 값을, 없으면 기본값을 돌려준다."""
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH, encoding='utf-8') as f:
            data = json.load(f)
        pts = data.get('src_points')
        if pts is not None:
            return np.array(pts, dtype=np.float32)
    return SRC_POINTS_DEFAULT.copy()


def save_calib(src_points, note=''):
    """튜닝된 SRC_POINTS를 calib.json 에 저장한다."""
    data = {
        'src_points': np.asarray(src_points, dtype=float).tolist(),
        'note': note,
    }
    with open(CALIB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return CALIB_PATH
