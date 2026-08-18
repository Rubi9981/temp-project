"""cross_road 주행 시스템 전역 설정 파일.

사용자가 손쉽게 모든 파라미터와 변수를 한 곳에서 확인하고 튜닝할 수 있도록
상단에 체계적으로 모아두었습니다.
"""
import json
import os
from dataclasses import dataclass

import numpy as np

# ==============================================================================
# 1. 경로 설정 (Path Settings)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)
LANE_DRIVE_DIR = os.path.join(WORKSPACE_DIR, 'lane-drive')
OBJECT_DETECTION_DIR = os.path.join(WORKSPACE_DIR, 'object_detection')
PROJECT_DIR = os.path.join(WORKSPACE_DIR, 'project')

CALIB_PATH = os.path.join(LANE_DRIVE_DIR, 'calib.json')
METRIC_PATH = os.path.join(LANE_DRIVE_DIR, 'metric.json')


# ==============================================================================
# 2. 주행 속도 설정 (Speed Settings)
# ==============================================================================
SPEED_CROSSROAD = 50       # 교차로 직진 통과 속도 (사용자 지정: 50)
SPEED_NORMAL = 100         # 일반 차선 추종 주행 속도
SPEED_MANUAL = 100         # 수동 조종(MANUAL) 모드 속도
SPEED_STOP = 0             # 정지 속도


# ==============================================================================
# 3. 조향 및 서보 모터 설정 (Steering & Servo Settings)
# ==============================================================================
SERVO_CENTER = 90          # 서보 중립 각도 (직진)
SERVO_MIN = 30             # 서보 최소 각도 (좌선회 최대)
SERVO_MAX = 150            # 서보 최대 각도 (우선회 최대)
SERVO_EMA_ALPHA = 0.5      # 서보 평활 지수이동평균 계수 (1.0 = 평활 없음)
INVERT_SERVO = False       # 서보 방향 반전 여부

# Pure Pursuit 제어기 파라미터
LOOKAHEAD_CM = 20.0        # 전방 주시 거리 (Lookahead distance, cm)
MAX_STEER_DEG = 28.0       # 앞바퀴 최대 조향각 (deg)
SERVO_PER_DEG = 60.0 / MAX_STEER_DEG  # 서보 단위 / 조향각(deg)
STEER_CLIP = 200           # 조향 오차 클립 한계


# ==============================================================================
# 4. 객체 탐지 설정 (Object Detection & YOLO Settings)
# ==============================================================================
# 사용자 지정 7대 클래스
TARGET_CLASSES = [
    'red',         # 적색 신호등
    'left',        # 좌회전 신호
    'right',       # 우회전 신호
    'car_red',     # 적색 차량 장애물
    'car_white',   # 백색 차량 장애물
    'human',       # 보행자
    'right_sign',  # 우회전 표지판
]

# 탐지 시 차량을 즉시 멈춰야 하는 안전 위험/정지 신호 클래스
SAFETY_STOP_CLASSES = ['red', 'human', 'car_red', 'car_white']

# YOLO 모델 경로 (NCNN 폴더 또는 .pt 파일)
# 라즈베리파이 가속용 NCNN 모델 기본 사용, 없을 경우 best_v3.pt 등으로 폴백
YOLO_MODEL_PATH = os.path.join(OBJECT_DETECTION_DIR, 'best_v3_ncnn_model')
if not os.path.exists(YOLO_MODEL_PATH):
    YOLO_MODEL_PATH = os.path.join(OBJECT_DETECTION_DIR, 'best_v3.pt')

YOLO_CONF = 0.25           # 객체 탐지 신뢰도 임계값
YOLO_IMGSZ = 640           # 추론 이미지 해상도
YOLO_EVERY = 5             # N프레임마다 YOLO 추론 (주행 루프 반응성 유지)


# ==============================================================================
# 5. 교차로 직진 판단 파라미터 (Crossroad Decision Settings)
# ==============================================================================
# 차선이 검출되지 않고 객체가 없을 때 교차로 직진 상태로 전환할 프레임 수
CROSSROAD_ENTER_FRAMES = 1

# 교차로 직진 주행 중 객체/장애물이 없을 때 안전을 보장하는 최대 직진 프레임 수 (초과 시 안전 정지)
CROSSROAD_MAX_FRAMES = 120  # 약 4초 (30fps 기준)

# 연속 실패로 완전 정지하기까지의 최대 프레임 수 (일반 주행 중 차선 상실 시)
MAX_FAIL_FRAMES = 10


# ==============================================================================
# 6. 카메라 및 영상 규격 (Camera & Vision Settings)
# ==============================================================================
W, H = 640, 480            # 카메라 해상도 (Width, Height)
CAMERA_FPS = 30            # 카메라 목표 FPS

# BEV (Bird's Eye View) 변환 기준 좌표 [좌상, 우상, 우하, 좌하]
SRC_POINTS_DEFAULT = np.array([
    [36, 262],   # 좌상단
    [588, 269],  # 우상단
    [605, 406],  # 우하단
    [17, 413],   # 좌하단
], dtype=np.float32)

# ROI (하단 관심 영역)
ROI_Y_RATIO = 0.55         # BEV 하단 ROI 시작 비율 (전체의 아래쪽 45% 사용)
MIN_AREA = 300             # 차선 컨투어 최소 면적
BLUR_KSIZE = (5, 5)
MORPH_KSIZE = (5, 5)
MORPH_ITER_OPEN = 1
MORPH_ITER_CLOSE = 2


# ==============================================================================
# 7. 이진화 및 차선 검출 파라미터 (Lane Detection Settings)
# ==============================================================================
# 기본 이진화 방식: 'adaptive' (또는 'tophat', 'hsv')
DEFAULT_BINARIZE = 'adaptive'
ADAPT_BLOCK = 21
ADAPT_C = -4

# HSV 기준값 (hsv 모드시)
LOWER_HSV = np.array([50, 0, 135], dtype=np.uint8)
UPPER_HSV = np.array([105, 50, 255], dtype=np.uint8)

# 슬라이딩 윈도우 파라미터
DEFAULT_DETECT = 'sliding'
SW_NWINDOWS = 9
SW_MARGIN = 60
SW_MINPIX = 50
SW_MIN_PEAK = 15
SW_MIN_FITPIX = 200
LANE_WIDTH_PX = 457.5


# ==============================================================================
# 8. 차량 제원 및 미터법 환산 (Metric & Vehicle Spec)
# ==============================================================================
LANE_WIDTH_CM_DEFAULT = 20.0
WHEELBASE_CM_DEFAULT = 11.0
PX_PER_CM_Y_DEFAULT = 12.0
REAR_AXLE_OFFSET_CM_DEFAULT = 12.0
VEHICLE_CENTER_X_PX_DEFAULT = 320.0


@dataclass(frozen=True)
class Metric:
    px_per_cm_x: float
    px_per_cm_y: float
    vehicle_center_x_px: float
    rear_axle_offset_cm: float
    wheelbase_cm: float
    lane_width_cm: float
    measured: bool
    note: str = ''


def get_metric():
    """metric.json 이 있으면 로드하고, 없으면 기본값을 반환합니다."""
    if os.path.exists(METRIC_PATH):
        with open(METRIC_PATH, encoding='utf-8') as f:
            data = json.load(f)
        return Metric(
            px_per_cm_x=data.get('px_per_cm_x', LANE_WIDTH_PX / LANE_WIDTH_CM_DEFAULT),
            px_per_cm_y=data.get('px_per_cm_y', PX_PER_CM_Y_DEFAULT),
            vehicle_center_x_px=data.get('vehicle_center_x_px', VEHICLE_CENTER_X_PX_DEFAULT),
            rear_axle_offset_cm=data.get('rear_axle_offset_cm', REAR_AXLE_OFFSET_CM_DEFAULT),
            wheelbase_cm=data.get('wheelbase_cm', WHEELBASE_CM_DEFAULT),
            lane_width_cm=data.get('lane_width_cm', LANE_WIDTH_CM_DEFAULT),
            measured=True,
            note=data.get('note', ''),
        )
    return Metric(
        px_per_cm_x=LANE_WIDTH_PX / LANE_WIDTH_CM_DEFAULT,
        px_per_cm_y=PX_PER_CM_Y_DEFAULT,
        vehicle_center_x_px=VEHICLE_CENTER_X_PX_DEFAULT,
        rear_axle_offset_cm=REAR_AXLE_OFFSET_CM_DEFAULT,
        wheelbase_cm=WHEELBASE_CM_DEFAULT,
        lane_width_cm=LANE_WIDTH_CM_DEFAULT,
        measured=False,
        note='기본 추정값',
    )


def get_src_points():
    """calib.json 이 있으면 로드하고, 없으면 기본값을 반환합니다."""
    if os.path.exists(CALIB_PATH):
        with open(CALIB_PATH, encoding='utf-8') as f:
            data = json.load(f)
        pts = data.get('src_points')
        if pts is not None:
            return np.array(pts, dtype=np.float32)
    return SRC_POINTS_DEFAULT.copy()
