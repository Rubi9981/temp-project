"""주행 상태 기계.

한 프레임을 받아 인지 파이프라인을 돌리고 조향/구동 명령까지 내린다.

    warp -> ROI -> 이진화 -> 검출 -> Pure Pursuit -> servo/motor

drive.py 에서 `import driver as driverlib` 로 쓴다 — main() 의 지역 변수
`driver` 와 이름이 겹치기 때문. bev.py 를 bevlib 로 받는 것과 같은 이유다.
"""
import threading

import numpy as np

import bev as bevlib
import config as cfg


class Driver:
    """주행 상태 기계.

    모드는 세 가지다.
        AUTO   — Pure Pursuit 자율주행
        MANUAL — 웹에서 방향키/버튼으로 직접 조종 (인지는 계속 돌아 화면은 살아 있다)
        STOP   — 모터 정지

    하드웨어는 주행 스레드와 웹 스레드가 함께 만지므로 hw_lock 으로 직렬화한다.
    """

    def __init__(self, hw, pp, bin_fn, det_fn, speed,
                 max_fail=None, ema_alpha=None, invert_servo=False,
                 manual_speed=None, mode='AUTO'):
        self.hw = hw
        self.pp = pp
        self.bin_fn = bin_fn
        self.det_fn = det_fn
        self.speed = speed
        self.max_fail = cfg.MAX_FAIL_FRAMES if max_fail is None else max_fail
        self.alpha = cfg.SERVO_EMA_ALPHA if ema_alpha is None else ema_alpha
        self.invert_servo = invert_servo
        self.manual_speed = cfg.DRIVE_SPEED if manual_speed is None else manual_speed

        self.hw_lock = threading.Lock()
        self.mode = mode                    # 'AUTO' | 'MANUAL' | 'STOP'
        self.servo_cmd = float(cfg.SERVO_CENTER)
        self.motor_cmd = 0
        self.fail_streak = 0
        self.stopped = False
        self.stats = {'frames': 0, 'ok': 0, 'fail': 0, 'halt': 0}

    # -----------------------------
    # 하드웨어 출력 (양쪽 스레드가 공유)
    # -----------------------------
    def apply_servo(self, angle):
        # 내부 상태는 실수로 둔다. 정수로 반올림해 보관하면 EMA 가 매 프레임
        # 양자화되어 평활 결과가 미세하게 달라진다.
        self.servo_cmd = float(np.clip(angle, cfg.SERVO_MIN, cfg.SERVO_MAX))
        with self.hw_lock:
            self.hw.servo(int(round(self.servo_cmd)))

    def apply_motor(self, speed):
        self.motor_cmd = int(speed)
        with self.hw_lock:
            self.hw.motor(int(speed))

    def set_mode(self, mode):
        """모드 전환. 어느 방향이든 일단 모터를 세우고 들어간다."""
        if mode not in ('AUTO', 'MANUAL', 'STOP'):
            return False
        self.mode = mode
        self.apply_motor(0)
        if mode == 'AUTO':
            self.fail_streak = 0
            self.stopped = False
        elif mode == 'MANUAL':
            self.apply_servo(cfg.SERVO_CENTER)
        return True

    def step(self, frame):
        """한 프레임 처리. (ctrl, warped, res, y_start) 를 돌려준다."""
        self.stats['frames'] += 1

        warped, _ = bevlib.warp_image(frame)
        roi, y_start = bevlib.roi_of(warped)
        mask = self.bin_fn(roi)
        res = self.det_fn(mask)
        ctrl = self.pp(res, roi.shape[0], y_start)

        if self.mode == 'STOP':
            # 인지는 계속 돌린다 — 화면은 살아 있어야 상태를 볼 수 있다
            self.apply_motor(0)
            return ctrl, warped, res, y_start

        if self.mode == 'MANUAL':
            # 조향/구동은 웹 핸들러가 직접 넣는다. 여기서는 건드리지 않는다.
            return ctrl, warped, res, y_start

        if ctrl.ok:
            self.stats['ok'] += 1
            self.fail_streak = 0
            self.stopped = False
            # 지수이동평균으로 프레임 간 튀는 명령을 완화한다
            target = ctrl.servo
            if self.invert_servo:
                target = 2 * cfg.SERVO_CENTER - target
            smoothed = self.servo_cmd + self.alpha * (target - self.servo_cmd)
            self.apply_servo(smoothed)
            self.apply_motor(self.speed)
        else:
            self.stats['fail'] += 1
            self.fail_streak += 1
            if self.fail_streak >= self.max_fail:
                # 차선을 계속 못 찾으면 직전 조향을 유지한 채 세운다.
                # 모르는 상태로 계속 달리는 것보다 안전하다.
                if not self.stopped:
                    self.stats['halt'] += 1
                    print(f'  [정지] 연속 {self.fail_streak}프레임 검출 실패')
                self.stopped = True
                self.apply_motor(0)
            else:
                # 짧은 끊김은 직전 명령을 유지하고 넘어간다
                self.apply_servo(self.servo_cmd)
                self.apply_motor(self.speed)

        return ctrl, warped, res, y_start
