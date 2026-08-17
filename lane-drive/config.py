"""모든 파라미터의 단일 소스.

기존 project/ 에서는 SRC_POINTS가 6개 파일에 복붙되어 있어 재튜닝 시
일부만 수정되면 조용히 어긋났다. 여기서는 이 파일 하나만 본다.

튜닝 결과는 calib.json 에 저장되며, 존재하면 기본값보다 우선한다.
"""
import json
import os
from dataclasses import dataclass

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

# ==============================================================================
# 미터법 환산 + 차량 제원 (Pure Pursuit)
# ==============================================================================
# 종방향 스케일과 뒤축 오프셋은 아직 미실측이다. calibrate_metric.py 로
# metric.json 을 만들면 그쪽이 우선한다.
METRIC_PATH = os.path.join(BASE_DIR, 'metric.json')

# gt.json 라벨 51쌍에서 잰 BEV 차선 폭 (띠 중심 <-> 띠 중심), std 25.8
GT_LANE_WIDTH_PX = 457.5

# --- 실측 완료 ---
LANE_WIDTH_CM_DEFAULT = 20.0        # 차선 띠 중심 <-> 띠 중심
WHEELBASE_CM_DEFAULT = 11.0         # 앞축 중심 <-> 뒤축 중심
# -> px_per_cm_x = 457.5 / 20 = 22.875 px/cm. BEV 가로 640px 는 실제 28.0cm.

# --- 미실측 (바닥 기준점 2개 촬영 후 calibrate_metric.py) ---
# 아래 두 값은 추정치다. 근거: 가로 가시폭이 28cm 로 나왔고 카메라가 낮게
# 전방을 보므로 종방향 가시 범위를 그보다 넓은 40cm 로 잡아 480/40 = 12.
# 실제 값은 기준점 2개를 찍어야 나온다. 그 전까지 servo 절대값은 신뢰하지 말 것.
PX_PER_CM_Y_DEFAULT = 12.0
REAR_AXLE_OFFSET_CM_DEFAULT = 12.0   # 뒤축 -> BEV 최하단 행이 비추는 지면까지
VEHICLE_CENTER_X_PX_DEFAULT = 320.0  # BEV상 차량 중심선. 기본은 화면 중앙

# -----------------------------
# 제어 파라미터
# -----------------------------
# look-ahead 통상 범위는 휠베이스의 1.5~3배. L=11cm 이므로 17~33cm.
# 짧으면 진동, 길면 코너 컷. 최종값은 실차 주행으로만 맞출 수 있다.
LOOKAHEAD_CM = 20.0
MAX_STEER_DEG = 28.0    # 앞바퀴 최대 조향각 — 실차 확인 필요

# 서보: 중립 90, 범위 30~150 (±60 대칭).
# raspi/L_5_Capture.py 기준 **왼쪽이 작은 값**(ArrowLeft -> servo(40)) 이므로
# 좌선회(delta > 0)일 때 servo 는 90보다 작아진다.
SERVO_CENTER = 90
SERVO_MIN = 30
SERVO_MAX = 150
# 서보 단위 / 실제 바퀴각(deg). 실차에서만 잴 수 있다.
# 기본값 근거: 서보 가동폭 ±60 단위가 최대 조향각 28도에 대응한다고 가정 -> 60/28.
SERVO_PER_DEG = 60.0 / MAX_STEER_DEG


# ==============================================================================
# 실시간 주행 루프 (drive.py)
# ==============================================================================
DRIVE_SPEED = 100        # afb1.gpio.motor() 값. raspi/L_6_CNN.py 가 주행에 쓰던 값
CAMERA_FPS = 30

# 연속 검출 실패를 몇 프레임까지 직전 명령으로 버틸지. 넘으면 모터를 세운다.
# 정적 평가에서 중심선 미산출이 74장 중 9장(12%)이었으므로 30fps 기준
# 초당 서너 프레임은 실패한다고 봐야 한다.
MAX_FAIL_FRAMES = 5

# 서보 명령 지수이동평균 계수. 1.0 이면 평활 없음(원시 명령 그대로).
SERVO_EMA_ALPHA = 0.5


# ==============================================================================
# 객체 탐지 (yolo.py) — drive.py --yolo 로 켠다. 기본은 꺼짐
# ==============================================================================
# .pt 는 .gitignore 에 걸려 추적되지 않는다. Pi 로는 따로 복사해야 한다.
YOLO_MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR),
                               'object_detection', 'best_v3.pt')
YOLO_CONF = 0.25

# 모델 학습값이 640 이다. 낮추면 빨라지지만 탐지를 잃는다 —
# 150장 기준 448 은 검출의 90%, 320 은 73% 만 남았다. Pi 가 느리면 그때 낮춘다.
YOLO_IMGSZ = 640

# N프레임마다 한 번만 추론한다. 주행 루프 안에서 동기로 돌기 때문에,
# 추론하는 프레임에서는 그 시간만큼 서보 갱신이 늦어진다.
YOLO_EVERY = 15


@dataclass(frozen=True)
class Metric:
    """BEV 픽셀 <-> 차량 좌표(cm) 환산 + 차량 제원."""
    px_per_cm_x: float
    px_per_cm_y: float
    vehicle_center_x_px: float
    rear_axle_offset_cm: float
    wheelbase_cm: float
    lane_width_cm: float
    measured: bool          # False = 실측 전 임시값
    note: str = ''


def default_metric():
    return Metric(
        px_per_cm_x=GT_LANE_WIDTH_PX / LANE_WIDTH_CM_DEFAULT,
        px_per_cm_y=PX_PER_CM_Y_DEFAULT,
        vehicle_center_x_px=VEHICLE_CENTER_X_PX_DEFAULT,
        rear_axle_offset_cm=REAR_AXLE_OFFSET_CM_DEFAULT,
        wheelbase_cm=WHEELBASE_CM_DEFAULT,
        lane_width_cm=LANE_WIDTH_CM_DEFAULT,
        measured=False,
        note='실측 전 임시값 — calibrate_metric.py 를 돌리세요',
    )


def get_metric():
    """metric.json 이 있으면 그 값을, 없으면 임시 기본값을 돌려준다."""
    if os.path.exists(METRIC_PATH):
        with open(METRIC_PATH, encoding='utf-8') as f:
            data = json.load(f)
        return Metric(
            px_per_cm_x=data['px_per_cm_x'],
            px_per_cm_y=data['px_per_cm_y'],
            vehicle_center_x_px=data['vehicle_center_x_px'],
            rear_axle_offset_cm=data['rear_axle_offset_cm'],
            wheelbase_cm=data['wheelbase_cm'],
            lane_width_cm=data['lane_width_cm'],
            measured=True,
            note=data.get('note', ''),
        )
    return default_metric()


def save_metric(metric, note=''):
    data = {k: getattr(metric, k) for k in
            ('px_per_cm_x', 'px_per_cm_y', 'vehicle_center_x_px',
             'rear_axle_offset_cm', 'wheelbase_cm', 'lane_width_cm')}
    data['note'] = note
    with open(METRIC_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return METRIC_PATH


def bev_key(name, already_bev):
    """파일명을 gt.json 의 키로 바꾼다.

    라벨은 captures_bev/ 기준으로 찍혀 있고 그 파일들은 save_bev.py 가
    'bev_' 접두사를 붙여 만든 것이다. captures/ 원본을 순회할 때는
    접두사를 붙여줘야 라벨이 매칭된다.
    """
    return name if already_bev else f'bev_{name}'


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
