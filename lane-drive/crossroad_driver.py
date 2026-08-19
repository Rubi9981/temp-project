"""교차로 직진 주행 상태 기계.

차선이 끊긴 구간을 서보 중립으로 천천히 직진해 빠져나간다.
`drive.py --crossroad` 로 켜며, 켜지 않으면 기존 `Driver` 가 그대로 쓰인다.

**Driver 를 상속한다.** 인지(warp -> 이진화 -> 검출 -> Pure Pursuit)와 하드웨어
출력(`apply_servo`/`apply_motor`/`set_mode`/`hw_lock`)은 부모 것을 그대로 쓰고,
`step()` 의 **판단 부분만** 바꾼다. 덕분에 `loop.run_loop` 이 둘을 구분하지 않고
돌릴 수 있고, 원격 탐지 워치독(`link_halt`)도 자동으로 따라온다.

AUTO 모드에서의 판단 순서:

    1. link_halt          원격 탐지 링크가 끊김            -> 정지
    2. 회전 진행 중       start_turn() 으로 시작됨          -> 고정 조향 원호
    3. 정지 대상 객체     human (CROSSROAD_STOP_CLASSES)    -> 정지
    4. ctrl.ok            차선 정상                        -> Pure Pursuit (speed)
    5. 차선 없음 + 객체 없음                               -> 직진 (servo 90, 저속)
    6. 그 외              차선 없음 + 객체 있음            -> 기존 실패 처리

2번이 3번보다 앞인 것은 의도적이다. 정지 대상이 화면에 남아 있어 기동이
중간에 멈추는 것을 막는다.

**4번의 진입 조건이 "차선이 안 보임"이라는 점을 알고 써야 한다.** 차선 실종은
교차로·급커브·반사광을 구분하지 못한다. obstacles/ 1046장(교차로가 없는
데이터)으로 replay 하면 88프레임이 이 상태로 분류되는데, 이는 그 데이터의
차선 검출 실패 수와 정확히 같다 — 즉 **검출 실패 전부가 교차로로 오인된다.**
정지선 검출과 표지판 래치가 들어가면 그쪽을 진입 조건으로 옮겨야 한다.
"""
import math

import config as cfg
from driver import Driver


class CrossroadDriver(Driver):
    """교차로 직진 + 객체 반응이 붙은 Driver.

    det 는 생성할 때 받아 둔다 — 그래야 step(frame) 시그니처가 Driver 와 같아져
    loop.run_loop 이 그대로 돌릴 수 있다. det 가 None 이면 객체 판단이 빠지고
    "차선 없으면 직진"만 남는다.
    """

    def __init__(self, *args, det=None, crossroad_speed=None, turn_speed=None,
                 target_classes=None, stop_classes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.det = det
        self.crossroad_speed = (cfg.CROSSROAD_SPEED if crossroad_speed is None
                                else crossroad_speed)
        self.turn_speed = cfg.TURN_SPEED if turn_speed is None else turn_speed
        self.target_classes = set(target_classes or cfg.CROSSROAD_TARGET_CLASSES)
        self.stop_classes = set(stop_classes or cfg.CROSSROAD_STOP_CLASSES)

        self.sub_state = 'INIT'         # loop.py 가 상태표/HUD 에 그대로 띄운다
        self.crossroad_frames = 0
        self.turn_side = None           # 'left' | 'right' | None(회전 아님)
        self.turn_start_n = 0
        self.turn_exit_run = 0
        self.turn_servo = self._turn_servo_table()
        self.stats.update({'lane_follow': 0, 'crossroad': 0, 'object_stop': 0,
                           'turn': 0})

    def _turn_servo_table(self):
        """좌/우 회전에 쓸 서보값을 시작할 때 한 번만 계산한다.

        차선이 없으면 밀 중심선 자체가 없으므로 목표점을 **만들어** 넣는다.
        전방 LOOKAHEAD_CM 지점에서 옆으로 TURN_OFFSET_PX 만큼 민 점이고,
        결과는 곡률이 고정된 원호다.

        Pure Pursuit 을 거치는 이유는 나중에 TURN_OFFSET_PX 를 실측 거리로
        바꾸는 것이 한 줄이 되게 하기 위해서다. 기본값(260px)에서는 최대
        조향으로 포화하므로 servo 는 좌 30 / 우 150 이 된다.
        """
        table = {}
        px_per_cm_x = self.pp.m.px_per_cm_x
        for side, sign in (('left', +1.0), ('right', -1.0)):
            # 차량 좌표는 좌측이 + 다 (bev.bev_to_vehicle 참조)
            Y = sign * cfg.TURN_OFFSET_PX / px_per_cm_x
            X = math.sqrt(max(cfg.LOOKAHEAD_CM ** 2 - Y * Y, 1e-9))
            value = self.pp.servo(self.pp.steer_angle(self.pp.curvature(X, Y)))
            if self.invert_servo:
                # 차선 추종 경로가 하는 것과 같은 뒤집기. 빠뜨리면 "직진은
                # 맞는데 회전만 반대로 도는" 증상이 나온다.
                value = 2 * cfg.SERVO_CENTER - value
            table[side] = int(value)
        return table

    def set_mode(self, mode):
        ok = super().set_mode(mode)
        if ok:
            self.crossroad_frames = 0
            self.turn_side = None       # 모드가 바뀌면 진행 중인 회전을 버린다
            self.sub_state = 'LANE_FOLLOW' if mode == 'AUTO' else mode
        return ok

    def start_turn(self, side):
        """회전 기동 시작. 웹의 TURN L / TURN R 버튼이 부른다.

        AUTO 에서만 받는다 — MANUAL 은 방향키가 같은 서보를 만지므로 충돌한다.
        """
        if side not in ('left', 'right') or self.mode != 'AUTO':
            return False
        self.turn_side = side
        self.turn_start_n = self.stats['frames']
        self.turn_exit_run = 0
        self.stopped = False
        self.sub_state = f'TURN_{side.upper()}'
        print(f'  [회전] {side} 시작 — servo {self.turn_servo[side]}')
        return True

    def _step_turn(self, res):
        """회전 중 한 프레임. 조향은 고정, 탈출은 차선 재획득으로 판정한다.

        TURN_MIN_FRAMES 동안은 탈출 조건을 아예 보지 않는다 — 차선이 아직
        보이는 상태에서 버튼을 누르면 첫 프레임부터 조건이 만족되어 회전이
        0.2초 만에 끝나버리기 때문이다.

        **탈출 판정에 ctrl.ok 를 쓰면 안 된다.** sliding_window 는 한쪽만
        잡히면 LANE_WIDTH_PX 로 반대쪽을 외삽해 status='single' 로 중심선을
        만들고, 그러면 ctrl.ok 가 참이 된다. 회전 도중 긴 가로선이 기울며
        세로에 가까워지는 순간 그걸 한쪽 차선으로 오인해 조기 탈출하고,
        차는 존재하지 않는 중심선을 향하게 된다.
        """
        self.stats['turn'] += 1
        # apply_servo 는 평활을 하지 않는다 (EMA 는 차선 추종 분기에서 계산된다).
        # 원호를 늦게 시작하면 반경이 커져 못 돌기 때문에 한 프레임에 넣는다.
        self.apply_servo(self.turn_servo[self.turn_side])
        self.apply_motor(self.turn_speed)

        held = self.stats['frames'] - self.turn_start_n
        if held < cfg.TURN_MIN_FRAMES:
            # 최소 회전 구간. 차선이 아직 보이는 상태에서 버튼을 눌렀거나
            # 회전 초반에 가로선을 차선으로 오인해도 여기서 걸러진다.
            return

        solid = (res.status == 'ok' and res.width is not None
                 and abs(res.width - cfg.LANE_WIDTH_PX) < cfg.TURN_WIDTH_TOL_PX)
        self.turn_exit_run = self.turn_exit_run + 1 if solid else 0

        if self.turn_exit_run >= cfg.TURN_EXIT_FRAMES:
            print(f'  [회전] {self.turn_side} 완료 — {held}프레임, 차선 재획득')
            self.turn_side = None
            self.sub_state = 'LANE_FOLLOW'
        elif held > cfg.TURN_TIMEOUT_FRAMES:
            print(f'  [정지] 회전 {held}프레임 초과 — 차선을 못 잡았습니다')
            self.turn_side = None
            self.stats['halt'] += 1
            self.stopped = True
            self.sub_state = 'TURN_TIMEOUT'
            self.apply_motor(0)

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

        # 회전은 정지 대상 객체 검사보다 **먼저** 본다. 신호등에서 좌/우회전할
        # 때 red 가 아직 화면에 남아 있어 기동이 중간에 멈추는 것을 막는다.
        if self.turn_side is not None:
            self._step_turn(res)
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
