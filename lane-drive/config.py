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
MAX_FAIL_FRAMES = 5

# 서보 명령 지수이동평균 계수. 1.0 이면 평활 없음(원시 명령 그대로).
SERVO_EMA_ALPHA = 0.5


# ==============================================================================
# 객체 탐지 (yolo.py) — drive.py --yolo 로 켠다. 기본은 꺼짐
# ==============================================================================
# NCNN 으로 내보낸 모델(폴더)을 쓴다. 현재 기본 모델은 best_v6.pt 를
# imgsz=640 으로 변환한 best_v6_ncnn_model 이다.
# 변환 명령: yolo export model=object_detection/best_v6.pt format=ncnn imgsz=640
# NCNN 폴더는 맥의 yolo_server.py 에 두면 되고, Pi 원격 모드에서는 모델 파일이
# 필요 없다. 다른 장비에서 직접 로컬 추론할 때는 폴더 전체를 복사한다(scp -r).
YOLO_MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR),
                               'object_detection', 'best_v6_ncnn_model')
YOLO_CONF = 0.25

# **NCNN 모델은 내보낼 때 크기가 고정된다.** 지금 모델은 640 으로 내보냈으므로
# 이 값도 640 이어야 한다. 448 로 쓰려면 imgsz=448 로 다시 내보내고 여기도 바꾼다.
# (실행 시 --imgsz 로 덮어써도 NCNN 모델에는 통하지 않는다)
YOLO_IMGSZ = 640

# N프레임마다 한 번만 추론한다.
#
# 기본값이 3 인 것은 **원격 추론이 기본**이기 때문이다 — Pi CPU 를 쓰지 않으므로
# 자주 제안해도 제어 루프가 굶지 않고, 밀리면 loop.Worker 가 알아서 버린다.
# 표본이 많을수록 좌/우 화살표 판정이 안정되므로 자주 보는 편이 이득이다.
#
# **로컬 추론(--yolo)으로 되돌릴 때는 --yolo-every 를 15 쯤으로 올릴 것.**
# Pi4 에서 매 3프레임 추론은 코어를 계속 물고 있게 된다.
YOLO_EVERY = 3


# ==============================================================================
# 원격 객체 탐지 (yolo_server.py + yolo_remote.py) — drive.py --yolo-remote
# ==============================================================================
# Pi4 로컬 추론이 느려서, 프레임을 맥으로 보내 거기서 추론하고 결과만 받는다.
# 전송 방식은 HTTP POST 다. 자세한 배경은 yolo_remote.py docstring 참조.
# 추론 서버(맥)의 주소. drive.py 는 인자 없이 실행하면 여기로 붙는다.
# 맥 주소가 바뀌면 이 줄만 고치면 된다 (--yolo-remote 로 그때그때 덮어쓸 수도 있다).
YOLO_REMOTE_HOST = '100.124.14.110'
YOLO_REMOTE_PORT = 5010
YOLO_REMOTE_DEFAULT = f'{YOLO_REMOTE_HOST}:{YOLO_REMOTE_PORT}'

# 전송용 JPEG 품질. obstacles 210장 측정 (무손실 대비):
#   품질 95  87.2KB  검출 100%  클래스집합 일치 97%
#   품질 85  46.4KB  검출 101%  일치 97%      <- 기본값
#   품질 75  33.9KB  검출  99%  일치 94%
#   품질 50  22.4KB  검출  98%  일치 93%
# 대역폭은 병목이 아니다 (초당 15회 추론해도 5Mbps). Pi4 의 JPEG 인코딩
# 시간이 문제가 되면 그때 75 로 낮춘다 — 검출 손실은 1% 다.
YOLO_JPEG_QUALITY = 85

# 한 번의 왕복을 기다리는 최대 시간(초). 넘으면 "결과 없음"으로 처리하고
# 다음 프레임으로 넘어간다. 절대 예외로 죽지 않는다.
YOLO_TIMEOUT_S = 0.2

# 유효한 탐지 결과가 이 시간(ms) 이상 없으면 모터를 세운다. 0 이면 끈다.
#
# **추론 시도 간격보다 충분히 커야 한다.** YOLO_EVERY=15, 17fps 면 시도 간격이
# 이미 0.9초라 임계를 1000ms 로 잡으면 정상 주행 중에도 깜빡인다.
# 원격 모드에서는 Pi CPU 를 안 쓰므로 --yolo-every 를 낮추는 쪽이 맞다.
YOLO_WATCHDOG_MS = 2000


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


# ==============================================================================
# 교차로 통과 (crossroad_driver.py) — drive.py --crossroad 로 켠다
# ==============================================================================
# 차선이 끊긴 구간을 서보 중립으로 천천히 직진해 빠져나간다.
CROSSROAD_SPEED = 50

# 교차로 직진을 몇 프레임까지 허용할지. 넘으면 세운다.
# 30fps 기준 약 4초. 이 값에 걸린다는 것은 "차선이 오래 안 보이는데 교차로도
# 아니었다"는 뜻이므로, 계속 직진하는 것보다 서는 편이 안전하다.
CROSSROAD_MAX_FRAMES = 120

# 이 중 하나라도 보이면 "교차로가 아니다"로 보고 직진 모드에 들어가지 않는다.
CROSSROAD_TARGET_CLASSES = ['red', 'left', 'right',
                            'car_red', 'car_white', 'human', 'right_sign']

# 이 중 하나라도 보이면 즉시 모터를 세운다.
#
# **거리 게이트가 없다.** 화면 어디에 얼마나 멀리 있든 잡히기만 하면 선다.
# 규정상 정적 장애물(car_*)은 회피 대상이지 정지 대상이 아니므로, 실주행
# 튜닝 때 bbox 면적 임계를 넣거나 이 목록에서 빼는 것을 먼저 검토할 것.
CROSSROAD_STOP_CLASSES = ['red', 'human', 'car_red', 'car_white']


# ==============================================================================
# 미션 상태 관리자 (mission.py) — drive.py --mission 으로 켠다
# ==============================================================================
# **아직 주행에 연결되어 있지 않다.** 상태 전이만 하고 화면에 띄운다.

# 상태 진입 면적 임계 (bbox 픽셀 면적). **면적이 곧 거리 대용이다** — 트랙
# 반대편에 작게 잡힌 표지판에 반응하지 않게 하는 것이 목적이다.
#
# 초기값 근거: obstacles/ 1046장을 best_v3 로 훑은 클래스별 면적 분포에서
# 중앙값 언저리를 잡았다 (p10 / p50 / p90):
#     right_sign  1481 / 2927 / 8038      human      7913 / 11058 / 19418
#     red          426 / 1144 /  1174     car_red     783 /  1970 /  7160
#     left         356 /  518 /   981     car_white  1402 /  3338 / 11595
#     right        585 / 1012 /  2281
#
# **전부 실트랙 튜닝 대상이다.** 위 분포는 best_v3 기준이고 지금 모델은 v6 이며,
# 무엇보다 "어느 거리에서 반응해야 하는가"는 주행 속도에 달렸다.
MISSION_AREA_ENTER = {
    'right_sign': 2500,
    'red': 1100,
    'left': 500,
    'right': 1000,
    'human': 11000,
    'car_red': 2000,
    'car_white': 3300,
}

# 직선으로 볼 곡률 상한 (1/cm). 이 값 미만이면 Observation.straight 가 참이 된다.
#
# **미측정이다.** 근거는 기하뿐이다 — delta = atan(L * kappa), L = 11cm 이므로
# 앞바퀴 3도가 kappa 0.0048 에 해당한다. "거의 안 꺾은 상태"를 3도로 본 값이다.
# 제대로 잡으려면 replay 로 ctrl.kappa 분포를 찍고 직선 구간의 p90 을 써야 한다.
MISSION_STRAIGHT_KAPPA = 0.005

# 복귀 규칙 STRAIGHT_N: 직선 차선이 이만큼 연속으로 잡히면 LANE_FOLLOW 로 돌아간다.
# 30fps 에서 0.33초.
MISSION_RETURN_FRAMES = 10

# 복귀 규칙 GONE: 진입시킨 객체가 이만큼 연속으로 안 보이면 돌아간다.
# 한 프레임 탐지 누락으로 회피가 중간에 끝나는 것을 막는 최소한의 방어다.
MISSION_GONE_FRAMES = 5

# 어떤 상태든 이만큼 머물면 HALT. 30fps 에서 5초.
# GONE 을 쓰는 상태에도 걸린다 — 차량이 영영 시야에 남으면 갇히기 때문이다.
MISSION_TIMEOUT_FRAMES = 150

# 탐지 결과가 이보다 묵으면 "확인 불가"로 본다 (없는 것으로 치지 않는다).
# 원격 추론은 --yolo-every 와 왕복 지연만큼 결과가 늦으므로, 그 지연을 넘어서는
# 값이어야 한다.
MISSION_MAX_AGE_MS = 500


# ==============================================================================
# 좌/우 회전 기동 (crossroad_driver.py) — 웹의 TURN L / TURN R 버튼으로 시작
# ==============================================================================
# T자 교차로에서는 양쪽 세로 차선이 동시에 사라지고 가로선만 남는다. 따라갈
# 중심선이 없으므로 목표점을 **만들어** Pure Pursuit 에 넣는다. 결과는 곡률이
# 고정된 원호다. 나중에 가로선까지의 거리를 실측해 넣으면 반경만 정확해진다.

# 목표점을 옆으로 미는 양 (BEV px). 좌회전이 +, 우회전이 - 방향이다.
#
#   N(px)   요청 반경   delta      servo (좌 / 우)
#     160     28.6cm    21.0도      45 / 135
#     200     22.9cm    25.7도      35 / 145
#     260        —      28.0도(포화) 30 / 150      <- 기본값
#
# **221 이상은 전부 최대 조향으로 포화한다** (SERVO_PER_DEG = 60/28 이라 최대
# 조향이 정확히 SERVO_MIN/SERVO_MAX 에 떨어진다). 포화 상태의 실제 원호 반경은
# 요청값이 아니라 L/tan(28도) = 20.7cm 다.
#
# 완만하게 만들 여지가 좁다 — 유효 조정 폭이 160~221 뿐이고 그 아래로 내리면
# 반경이 급격히 커진다 (120px -> 38.1cm).
TURN_OFFSET_PX = 260

# 회전 중 모터 속도. CROSSROAD_SPEED 와 분리한다 — 직진 통과와 회전은 요구
# 속도가 다르고, 느릴수록 탈출 조건을 잡을 프레임이 늘어난다.
TURN_SPEED = 40

# 이만큼은 무조건 돈다 — 탈출 조건을 아예 보지 않는다.
#
# **없으면 회전이 즉시 끝난다.** 차선이 아직 보이는 상태에서 버튼을 누르면
# 탈출 조건이 첫 프레임부터 만족되어 5프레임(0.17초) 만에 복귀한다. 회전
# 초반에 가로선을 차선으로 잠깐 오인하는 경우도 같은 방식으로 막힌다.
# 30fps 기준 0.67초.
TURN_MIN_FRAMES = 20

# 양쪽 차선이 이만큼 연속으로 잡히면 회전을 끝내고 차선 추종으로 돌아간다.
TURN_EXIT_FRAMES = 5

# 탈출 판정의 차선 폭 상식 검사. gt.json 실측 mean 457.5 / std 25.8 의 약 3시그마.
# 회전 중 기울어진 가로선을 한쪽 차선으로 오인하는 것을 걸러낸다.
TURN_WIDTH_TOL_PX = 80

# 이만큼 돌았는데도 차선을 못 잡으면 정지 (30fps 기준 4초).
TURN_TIMEOUT_FRAMES = 120
