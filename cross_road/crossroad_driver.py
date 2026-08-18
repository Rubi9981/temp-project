"""교차로 직진 주행 상태 머신 (CrossroadDriver).

차선 검출 결과와 YOLO 객체 탐지 결과를 종합하여 다음 4가지 주행 모드를 제어합니다:
1. LANE_FOLLOW: 차선 정상 검출 -> Pure Pursuit 자율 주행 (설정 속도)
2. CROSSROAD_STRAIGHT: 차선 미검출 + 7종 객체 미탐지 -> 직진 주행 (speed=50, servo=90)
3. OBJECT_STOP: 정지 유발 객체(red, human, car_red, car_white) 탐지 -> 일시 정지 (speed=0)
4. FAIL_SAFE: 장시간 차선 상실 / 비정상 상태 -> 안전 정지 (speed=0)
"""
import sys
import os
import threading
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(CURRENT_DIR)
LANE_DRIVE_DIR = os.path.join(WORKSPACE_DIR, 'lane-drive')

for p in (CURRENT_DIR, LANE_DRIVE_DIR, WORKSPACE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import bev as bevlib

try:
    from cross_road import config as cfg
except ImportError:
    import config as cfg


class CrossroadDriver:
    """교차로 직진 및 객체 감지 통합 주행 제어기."""

    def __init__(self, hw, pp, bin_fn, det_fn,
                 speed_normal=cfg.SPEED_NORMAL,
                 speed_crossroad=cfg.SPEED_CROSSROAD,
                 target_classes=None,
                 safety_stop_classes=None,
                 servo_center=cfg.SERVO_CENTER,
                 max_fail=cfg.MAX_FAIL_FRAMES,
                 ema_alpha=cfg.SERVO_EMA_ALPHA,
                 invert_servo=cfg.INVERT_SERVO,
                 manual_speed=cfg.SPEED_MANUAL,
                 mode='AUTO'):
        self.hw = hw
        self.pp = pp
        self.bin_fn = bin_fn
        self.det_fn = det_fn

        self.speed_normal = speed_normal
        self.speed_crossroad = speed_crossroad
        self.target_classes = set(target_classes or cfg.TARGET_CLASSES)
        self.safety_stop_classes = set(safety_stop_classes or cfg.SAFETY_STOP_CLASSES)

        self.servo_center = servo_center
        self.max_fail = max_fail
        self.alpha = ema_alpha
        self.invert_servo = invert_servo
        self.manual_speed = manual_speed

        # BEV 변환 행렬 캐싱 (ROI 크기만 변환하여 연산 최적화)
        self.roi_matrix, self.y_start, self.roi_h = bevlib.roi_warp_matrix()

        self.hw_lock = threading.Lock()
        self.mode = mode                     # 'AUTO' | 'MANUAL' | 'STOP'
        self.sub_state = 'INIT'              # 'LANE_FOLLOW' | 'CROSSROAD_STRAIGHT' | 'OBJECT_STOP' | 'FAIL_SAFE'
        self.servo_cmd = float(self.servo_center)
        self.motor_cmd = 0
        self.fail_streak = 0
        self.crossroad_frames = 0
        self.stopped = False
        self.stats = {
            'frames': 0,
            'lane_follow': 0,
            'crossroad': 0,
            'object_stop': 0,
            'fail': 0,
            'halt': 0,
        }

    # --------------------------------------------------------------------------
    # 하드웨어 제어
    # --------------------------------------------------------------------------
    def apply_servo(self, angle):
        """서보 모터 각도 적용 (클립 및 평활)."""
        self.servo_cmd = float(np.clip(angle, cfg.SERVO_MIN, cfg.SERVO_MAX))
        with self.hw_lock:
            self.hw.servo(int(round(self.servo_cmd)))

    def apply_motor(self, speed):
        """구동 모터 속도 적용."""
        self.motor_cmd = int(speed)
        with self.hw_lock:
            self.hw.motor(int(speed))

    def set_mode(self, mode):
        """전체 주행 모드 전환 ('AUTO', 'MANUAL', 'STOP')."""
        if mode not in ('AUTO', 'MANUAL', 'STOP'):
            return False
        self.mode = mode
        self.apply_motor(0)
        if mode == 'AUTO':
            self.fail_streak = 0
            self.crossroad_frames = 0
            self.stopped = False
            self.sub_state = 'LANE_FOLLOW'
        elif mode == 'MANUAL':
            self.apply_servo(self.servo_center)
            self.sub_state = 'MANUAL'
        return True

    # --------------------------------------------------------------------------
    # 프레임 처리 및 주행 판단
    # --------------------------------------------------------------------------
    def step(self, frame, det=None):
        """한 프레임을 처리하고 주행 명령을 내립니다.
        
        Args:
            frame: BGR 입력 영상 (640x480)
            det: YOLO Detector 인스턴스 (또는 None)
            
        Returns:
            (ctrl, roi, res, y_start, sub_state)
        """
        self.stats['frames'] += 1

        # 1. BEV Warp 및 차선 검출 파이프라인
        roi = cv2.warpPerspective(frame, self.roi_matrix, (cfg.W, self.roi_h))
        y_start = self.y_start
        mask = self.bin_fn(roi)
        res = self.det_fn(mask)
        ctrl = self.pp(res, roi.shape[0], y_start)

        # 2. 객체 탐지 결과 분석
        detected_counts = det.counts if (det is not None and hasattr(det, 'counts')) else {}
        
        # 화면에 7대 대상 클래스가 존재하는지 확인
        active_targets = [cls for cls in self.target_classes if detected_counts.get(cls, 0) > 0]
        has_target = len(active_targets) > 0

        # 안전 정지 클래스(red, human, car_red, car_white) 존재 확인
        active_safety_stops = [cls for cls in self.safety_stop_classes if detected_counts.get(cls, 0) > 0]
        has_safety_stop = len(active_safety_stops) > 0

        # 3. 주행 모드별 분기
        if self.mode == 'STOP':
            self.sub_state = 'STOP'
            self.apply_motor(0)
            return ctrl, roi, res, y_start, self.sub_state

        if self.mode == 'MANUAL':
            self.sub_state = 'MANUAL'
            return ctrl, roi, res, y_start, self.sub_state

        # ======================================================================
        # [AUTO 모드 주행 의사결정 로직]
        # ======================================================================

        # Case 1: 안전 위험 객체 탐지 시 (적색 신호, 보행자, 장애물 차량) -> 정지
        if has_safety_stop:
            self.sub_state = f'STOP_{active_safety_stops[0].upper()}'
            self.stats['object_stop'] += 1
            self.apply_motor(cfg.SPEED_STOP)
            # 서보는 직진 유지
            self.apply_servo(self.servo_center)
            return ctrl, roi, res, y_start, self.sub_state

        # Case 2: 차선이 정상 검출된 경우 -> 일반 차선 추종 주행
        if ctrl.ok:
            self.sub_state = 'LANE_FOLLOW'
            self.stats['lane_follow'] += 1
            self.fail_streak = 0
            self.crossroad_frames = 0
            self.stopped = False

            # Pure Pursuit 조향각 계산 및 EMA 평활
            target_servo = ctrl.servo
            if self.invert_servo:
                target_servo = 2 * self.servo_center - target_servo
            smoothed_servo = self.servo_cmd + self.alpha * (target_servo - self.servo_cmd)

            self.apply_servo(smoothed_servo)
            self.apply_motor(self.speed_normal)
            return ctrl, roi, res, y_start, self.sub_state

        # Case 3: 차선이 보이지 않고(ctrl.ok=False) + 7개 객체도 탐지되지 않은 경우
        # -> 사용자가 요구한 [교차로 직진 알고리즘] 적용!
        if not ctrl.ok and not has_target:
            self.sub_state = 'CROSSROAD_STRAIGHT'
            self.stats['crossroad'] += 1
            self.crossroad_frames += 1
            self.fail_streak = 0
            self.stopped = False

            # 최대 허용 교차로 통과 프레임 이내에서는 천천히 직진 (speed=50, servo=90)
            if self.crossroad_frames <= cfg.CROSSROAD_MAX_FRAMES:
                # 조향각은 직진(중립)으로 점진적 정렬 또는 즉시 90 설정
                straight_servo = self.servo_cmd + self.alpha * (self.servo_center - self.servo_cmd)
                self.apply_servo(straight_servo)
                self.apply_motor(self.speed_crossroad)  # 속도 50으로 직진
            else:
                # 교차로 직진이 너무 오래 지속되면 안전을 위해 정지
                self.sub_state = 'CROSSROAD_TIMEOUT_STOP'
                self.apply_motor(0)
                self.stopped = True

            return ctrl, roi, res, y_start, self.sub_state

        # Case 4: 차선이 안 보이는데, 다른 객체(예: 표지판 등)가 있거나 정의되지 않은 상태
        self.stats['fail'] += 1
        self.fail_streak += 1
        if self.fail_streak >= self.max_fail:
            self.sub_state = 'FAIL_SAFE_STOP'
            if not self.stopped:
                self.stats['halt'] += 1
                print(f'  [안전 정지] 연속 {self.fail_streak}프레임 검출 실패 (객체: {active_targets})')
            self.stopped = True
            self.apply_motor(0)
        else:
            self.sub_state = f'HOLD_LANE_LOST_{self.fail_streak}'
            # 짧은 끊김은 직전 조향 유지, 저속 주행
            self.apply_servo(self.servo_cmd)
            self.apply_motor(self.speed_crossroad)

        return ctrl, roi, res, y_start, self.sub_state
