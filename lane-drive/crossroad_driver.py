"""교차로 직진 주행 상태 기계.

차선이 끊긴 구간을 서보 중립으로 천천히 직진해 빠져나간다.
`drive.py --crossroad` 로 켜며, 켜지 않으면 기존 `Driver` 가 그대로 쓰인다.

**Driver 를 상속한다.** 인지(warp -> 이진화 -> 검출 -> Pure Pursuit)와 하드웨어
출력(`apply_servo`/`apply_motor`/`set_mode`/`hw_lock`)은 부모 것을 그대로 쓰고,
`step()` 의 **판단 부분만** 바꾼다. 덕분에 `loop.run_loop` 이 둘을 구분하지 않고
돌릴 수 있고, 원격 탐지 워치독(`link_halt`)도 자동으로 따라온다.

AUTO 모드에서의 판단 순서:

    1. link_halt          원격 탐지 링크가 끊김            -> 정지
    2. 정지 대상 객체     red / human / car_red / car_white -> 정지
    3. ctrl.ok            차선 정상                        -> Pure Pursuit (speed)
    4. 차선 없음 + 객체 없음                               -> 직진 (servo 90, 저속)
    5. 그 외              차선 없음 + 객체 있음            -> 기존 실패 처리

**4번의 진입 조건이 "차선이 안 보임"이라는 점을 알고 써야 한다.** 차선 실종은
교차로·급커브·반사광을 구분하지 못한다. obstacles/ 1046장(교차로가 없는
데이터)으로 replay 하면 88프레임이 이 상태로 분류되는데, 이는 그 데이터의
차선 검출 실패 수와 정확히 같다 — 즉 **검출 실패 전부가 교차로로 오인된다.**
정지선 검출과 표지판 래치가 들어가면 그쪽을 진입 조건으로 옮겨야 한다.
"""
import config as cfg
from driver import Driver


class CrossroadDriver(Driver):
    """교차로 직진 + 객체 반응이 붙은 Driver.

    det 는 생성할 때 받아 둔다 — 그래야 step(frame) 시그니처가 Driver 와 같아져
    loop.run_loop 이 그대로 돌릴 수 있다. det 가 None 이면 객체 판단이 빠지고
    "차선 없으면 직진"만 남는다.
    """

    def __init__(self, *args, det=None, crossroad_speed=None,
                 target_classes=None, stop_classes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.det = det
        self.crossroad_speed = (cfg.CROSSROAD_SPEED if crossroad_speed is None
                                else crossroad_speed)
        self.target_classes = set(target_classes or cfg.CROSSROAD_TARGET_CLASSES)
        self.stop_classes = set(stop_classes or cfg.CROSSROAD_STOP_CLASSES)

        self.sub_state = 'INIT'         # loop.py 가 상태표/HUD 에 그대로 띄운다
        self.crossroad_frames = 0
        self.stats.update({'lane_follow': 0, 'crossroad': 0, 'object_stop': 0})

    def set_mode(self, mode):
        ok = super().set_mode(mode)
        if ok:
            self.crossroad_frames = 0
            self.sub_state = 'LANE_FOLLOW' if mode == 'AUTO' else mode
        return ok

    def _seen(self, classes):
        """지금 보이는 것 중 classes 에 드는 이름들. det 가 없으면 빈 목록."""
        counts = getattr(self.det, 'counts', None) or {}
        return [c for c in classes if counts.get(c, 0) > 0]

    def step(self, frame):
        self.stats['frames'] += 1
        ctrl, roi, res, y_start = self.perceive(frame)

        if self.mode == 'STOP':
            self.sub_state = 'STOP'
            self.apply_motor(0)
            return ctrl, roi, res, y_start

        if self.mode == 'MANUAL':
            self.sub_state = 'MANUAL'
            return ctrl, roi, res, y_start

        # 원격 탐지 링크가 끊겼다. 객체 판단 자체를 믿을 수 없으므로
        # 아래 분기를 타기 전에 선다 — "안 보인다"와 "못 받았다"는 다르다.
        if self.link_halt:
            if not self.stopped:
                self.stats['halt'] += 1
                print('  [정지] 원격 탐지 링크 끊김')
            self.sub_state = 'LINK_LOST'
            self.stopped = True
            self.apply_motor(0)
            return ctrl, roi, res, y_start

        stopper = self._seen(self.stop_classes)
        if stopper:
            self.sub_state = f'STOP_{stopper[0].upper()}'
            self.stats['object_stop'] += 1
            self.apply_motor(0)
            self.apply_servo(cfg.SERVO_CENTER)
            return ctrl, roi, res, y_start

        if ctrl.ok:
            self.sub_state = 'LANE_FOLLOW'
            self.stats['lane_follow'] += 1
            self.stats['ok'] += 1
            self.fail_streak = 0
            self.crossroad_frames = 0
            self.stopped = False
            target = ctrl.servo
            if self.invert_servo:
                target = 2 * cfg.SERVO_CENTER - target
            self.apply_servo(self.servo_cmd + self.alpha * (target - self.servo_cmd))
            self.apply_motor(self.speed)
            return ctrl, roi, res, y_start

        # 차선이 없는데 대상 객체도 안 보인다 -> 교차로로 보고 직진한다
        if not self._seen(self.target_classes):
            self.crossroad_frames += 1
            self.fail_streak = 0
            if self.crossroad_frames > cfg.CROSSROAD_MAX_FRAMES:
                # 이만큼 직진했는데도 차선이 안 잡히면 교차로가 아니었던 것이다
                self.sub_state = 'CROSSROAD_TIMEOUT'
                if not self.stopped:
                    self.stats['halt'] += 1
                    print(f'  [정지] 교차로 직진 {self.crossroad_frames}프레임 초과')
                self.stopped = True
                self.apply_motor(0)
                return ctrl, roi, res, y_start

            self.sub_state = 'CROSSROAD_STRAIGHT'
            self.stats['crossroad'] += 1
            self.stopped = False
            # 중립으로 한 번에 꺾지 않고 EMA 로 밀어 넣는다 — 직전 조향이
            # 크게 꺾여 있었다면 즉시 90 으로 튀는 것이 오히려 위험하다
            self.apply_servo(self.servo_cmd
                             + self.alpha * (cfg.SERVO_CENTER - self.servo_cmd))
            self.apply_motor(self.crossroad_speed)
            return ctrl, roi, res, y_start

        # 차선도 없고 대상 객체는 보인다 — 교차로가 아니다. 기존 실패 처리.
        self.stats['fail'] += 1
        self.fail_streak += 1
        if self.fail_streak >= self.max_fail:
            self.sub_state = 'FAIL_SAFE'
            if not self.stopped:
                self.stats['halt'] += 1
                print(f'  [정지] 연속 {self.fail_streak}프레임 검출 실패')
            self.stopped = True
            self.apply_motor(0)
        else:
            self.sub_state = f'HOLD_{self.fail_streak}'
            self.apply_servo(self.servo_cmd)
            self.apply_motor(self.speed)

        return ctrl, roi, res, y_start
