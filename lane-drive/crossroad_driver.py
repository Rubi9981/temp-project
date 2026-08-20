"""교차로 통과 · 좌우 회전 · 신호등/표지판 자동 반응 주행 상태 기계.

차선이 끊긴 구간을 서보 중립으로 천천히 직진해 빠져나가고, 고정 조향 원호로
좌/우 회전하며, 탐지한 객체에 따라 감속·정지·회전을 스스로 시작한다.
**기본으로 켜져 있다** — `drive.py --no-crossroad` 를 주면 기존 `Driver` 가 쓰인다.

**Driver 를 상속한다.** 인지(warp -> 이진화 -> 검출 -> Pure Pursuit)와 하드웨어
출력(`apply_servo`/`apply_motor`/`set_mode`/`hw_lock`)은 부모 것을 그대로 쓰고,
`step()` 의 **판단 부분만** 바꾼다. 덕분에 `loop.run_loop` 이 둘을 구분하지 않고
돌릴 수 있고, 원격 탐지 워치독(`link_halt`)도 자동으로 따라온다.

AUTO 모드에서의 판단 순서:

    1. link_halt          원격 탐지 링크가 끊김            -> 정지
    2. 회전 진행 중       start_turn() 으로 시작됨          -> 고정 조향 원호
    3. 정지 대상 객체     human (CROSSROAD_STOP_CLASSES)    -> 정지
    4. 자동 회전 트리거   화살표·표지판 면적 >= 임계        -> 회전 시작
    5. 빨간불 면적 초과   DETECTION_AREA_ENTER['red']         -> 정지 후 대기
    6. 정적 장애물 회피   car_white -> 오른쪽 / car_red -> 왼쪽 (중심선 이동)
    7. ctrl.ok            차선 정상 (감속이 켜져 있으면 x0.5) -> Pure Pursuit
    8. 차선 없음 + 객체 없음                               -> 직진 (servo 90, 저속)
    9. 그 외              차선 없음 + 객체 있음            -> 기존 실패 처리

순서에 근거가 있다.

    2번이 3번보다 앞 — 정지 대상이 화면에 남아 기동이 중간에 멈추는 것을 막는다.
    3번이 4번보다 앞 — 사람이 앞을 막고 있으면 회전을 시작하지 않는다.
    4번이 5번보다 앞 — 빨간불에 서 있다가 화살표가 뜨면 그때 회전해야 한다.
    6번이 7번 바로 앞 — 회피는 차선 추종의 변형이다 (중심선만 옆으로 민다).

4·5·6번의 자동 반응은 **기본으로 켜져 있다.** drive.py 의 --no-slow-on-sight /
--no-red-stop / --no-auto-turn 으로 각각 끈다.

**8번(교차로 직진)의 진입 조건이 "차선이 안 보임"이라는 점을 알고 써야 한다.**
차선 실종은 교차로·급커브·반사광을 구분하지 못한다. images/obstacles(교차로가
없는 데이터)로 replay 하면 88프레임이 이 상태로 분류되는데, 이는 그 데이터의
차선 검출 실패 수와 정확히 같다 — 즉 **검출 실패 전부가 교차로로 오인된다.**
정지선 검출이 들어가면 그쪽을 진입 조건으로 옮겨야 한다.
"""
import collections
import math

import config as cfg
import detect
import yolo
from driver import Driver


class CrossroadDriver(Driver):
    """교차로 직진 + 객체 반응이 붙은 Driver.

    det 는 생성할 때 받아 둔다 — 그래야 step(frame) 시그니처가 Driver 와 같아져
    loop.run_loop 이 그대로 돌릴 수 있다. det 가 None 이면 객체 판단이 빠지고
    "차선 없으면 직진"만 남는다.
    """

    def __init__(self, *args, det=None, crossroad_speed=None, turn_speed=None,
                 avoid_speed=None, target_classes=None, stop_classes=None,
                 slow_on_sight=True, red_stop=True, auto_turn=True, avoid=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.det = det
        # 자동 반응 세 가지. **기본으로 켜져 있다** (drive.py 의 --no-* 로 끈다).
        self.slow_on_sight = slow_on_sight
        self.red_stop = red_stop
        self.auto_turn = auto_turn
        self.avoid = avoid
        # 면적 임계는 **DETECTION_AREA_ENTER 를 그대로 읽는다.** 사용자가 실측해
        # 넣은 값이고, 별도 상수를 두면 같은 물리량을 두 곳에서 튜닝하게 된다.
        self.area_enter = dict(cfg.DETECTION_AREA_ENTER)
        self.crossroad_speed = (cfg.CROSSROAD_SPEED if crossroad_speed is None
                                else crossroad_speed)
        self.turn_speed = cfg.TURN_SPEED if turn_speed is None else turn_speed
        self.avoid_speed = cfg.AVOID_SPEED if avoid_speed is None else avoid_speed
        self.target_classes = set(target_classes or cfg.CROSSROAD_TARGET_CLASSES)
        self.stop_classes = set(stop_classes or cfg.CROSSROAD_STOP_CLASSES)

        self.sub_state = 'INIT'         # loop.py 가 상태표/HUD 에 그대로 띄운다
        self.reason = '시작'            # 그 상태로 간 근거. 상태표에 함께 나간다
        # 판단 이력. **화면의 sub_state 는 현재 프레임 한 줄뿐이라, 트랙에서
        # 차가 이상하게 굴고 나서 보면 라벨이 이미 바뀌어 있다.** 무엇이 언제
        # 왜 바뀌었는지는 여기에만 남는다.
        self.history = collections.deque(maxlen=24)
        self.crossroad_frames = 0
        self.turn_side = None           # 'left' | 'right' | None(회전 아님)
        self.turn_start_n = 0
        self.turn_exit_run = 0
        self.turn_back_left = 0     # 좌회전 시작 전 남은 후진 프레임 수
        # 회전이 끝난 프레임. 쿨다운 기준점이다. 시작할 때는 쿨다운이 걸려
        # 있으면 안 되므로 과거로 밀어 둔다.
        self.turn_done_n = -cfg.TURN_COOLDOWN_FRAMES
        self.avoid_side = None          # 'left' | 'right' | None(회피 아님)
        self.avoid_start_n = 0
        # 방향별 회피 종료 프레임. 쿨다운 기준점이라 시작할 때는 걸려 있으면
        # 안 되므로 과거로 밀어 둔다. **방향을 나눠 세는 것이 요점이다** —
        # 오른쪽 회피 직후에도 왼쪽 회피는 즉시 걸려야 한다.
        self.avoid_done_n = {'left': -cfg.AVOID_COOLDOWN_FRAMES,
                             'right': -cfg.AVOID_COOLDOWN_FRAMES}
        self.turn_servo = self._turn_servo_table()
        for side in ('left', 'right'):
            mn, to = self._turn_frames(side)
            if to <= mn:
                # 조용히 실패하는 조합이라 시작할 때 잡는다 — 탈출 조건을 보기
                # 시작하는 시점이 이미 타임아웃을 넘으면 그 방향은 매번 정지한다.
                print(f'[경고] {side} 회전: TURN_TIMEOUT({to}) <= TURN_MIN({mn}) 이라 '
                      '항상 타임아웃으로 끝납니다.')
        self.stats.update({'lane_follow': 0, 'crossroad': 0, 'object_stop': 0,
                           'turn': 0, 'avoid': 0})

    def _set_state(self, name, reason=''):
        """판단 결과를 기록한다. 이름이 바뀔 때만 이력에 남긴다.

        매 프레임 같은 상태를 다시 넣어도 이력이 더러워지지 않는다. 그래서
        면적처럼 매 프레임 달라지는 값은 **이름이 아니라 reason 에** 넣는다 —
        이름에 넣으면 프레임마다 새 전이로 잡힌다.
        """
        if name != self.sub_state:
            self.history.append((self.stats['frames'], name, reason))
        self.sub_state = name
        self.reason = reason

    @property
    def status_str(self):
        """상태표/HUD 에 나가는 한 줄. 상태와 근거를 함께 보여준다."""
        return f'{self.sub_state} — {self.reason}' if self.reason else self.sub_state

    def format_history(self, limit=None):
        """최근 판단 이력. 웹 화면은 limit 로 잘라 쓰고, 종료 요약은 전부 쓴다.

        웹에서 전부 띄우면 화면이 계속 길어져 스크롤하게 된다 — 주행 중에
        스크롤할 손이 없으므로 최근 몇 줄만 보여주는 편이 낫다.
        """
        if not self.history:
            return '(판단 전이 없음)'
        rows = list(self.history)[-limit:] if limit else list(self.history)
        return '\n'.join(f'[{f:6d}] {name:18s} {why}'
                         for f, name, why in rows)

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
            delta = self.pp.steer_angle(self.pp.curvature(X, Y))
            if self.invert_servo:
                # 차선 추종 경로가 하는 것과 같은 뒤집기. 빠뜨리면 "직진은
                # 맞는데 회전만 반대로 도는" 증상이 나온다.
                # **각도를 뒤집고 매핑한다** — 서보값을 반사하면 좌우 비대칭
                # 때문에 각도가 틀어진다 (servo() docstring 참조).
                delta = -delta
            table[side] = int(self.pp.servo(delta))
        return table

    def _turn_frames(self, side):
        """그 방향의 (최소 회전 프레임, 타임아웃 프레임).

        좌회전은 서보 좌측 가동각이 작아 반경이 크고, 그만큼 오래 걸린다.
        **둘 다 같은 배율로 늘려야 한다** — MIN 만 늘리면 탈출 조건을 보기
        시작하는 시점이 이미 타임아웃을 넘어 매번 정지로 끝난다.
        """
        scale = cfg.TURN_LEFT_FRAME_SCALE if side == 'left' else 1.0
        return (int(cfg.TURN_MIN_FRAMES * scale),
                int(cfg.TURN_TIMEOUT_FRAMES * scale))

    def set_mode(self, mode):
        ok = super().set_mode(mode)
        if ok:
            self.crossroad_frames = 0
            self.turn_side = None       # 모드가 바뀌면 진행 중인 회전을 버린다
            self.turn_back_left = 0
            self.avoid_side = None      # 모드가 바뀌면 진행 중인 회피도 버린다
            self._set_state('LANE_FOLLOW' if mode == 'AUTO' else mode,
                            f'모드 전환 -> {mode}')
        return ok

    def start_turn(self, side, auto=False):
        """회전 기동 시작. 웹의 TURN L / TURN R 버튼과 자동 트리거가 부른다.

        AUTO 에서만 받는다 — MANUAL 은 방향키가 같은 서보를 만지므로 충돌한다.
        쿨다운은 자동 트리거(_auto_turn_side) 쪽에서만 본다. **수동 버튼은
        쿨다운을 무시한다** — 사람이 보고 누른 것이므로 막을 이유가 없다.
        """
        if side not in ('left', 'right') or self.mode != 'AUTO':
            return False
        self.turn_side = side
        self.turn_start_n = self.stats['frames']
        self.turn_exit_run = 0
        self.stopped = False
        # **좌회전만 후진으로 시작한다.** 서보의 좌/우 가동각이 달라 좌회전
        # 반경이 크기 때문에, 물러나서 여유 거리를 만든 뒤 꺾는다.
        self.turn_back_left = cfg.TURN_BACK_FRAMES if side == 'left' else 0
        mn, to = self._turn_frames(side)
        why = f"{'자동' if auto else '수동'} 트리거, servo {self.turn_servo[side]}"
        self._set_state('TURN_LEFT_BACK' if self.turn_back_left
                        else f'TURN_{side.upper()}', why)
        back = (f'후진 {self.turn_back_left}프레임 후 ' if self.turn_back_left else '')
        print(f"  [회전] {side} 시작 ({'자동' if auto else '수동'}) — "
              f'{back}servo {self.turn_servo[side]}  최소 {mn} / 최대 {to}프레임')
        return True

    # -----------------------------
    # 자동 반응 (--slow-on-sight / --red-stop / --auto-turn)
    # -----------------------------
    def _areas(self):
        """지금 보이는 것들의 {클래스: (면적, conf)}. 면적이 곧 거리 대용이다."""
        return yolo.largest_area_by_class(getattr(self.det, 'boxes', None))

    def _slow_seen(self, areas):
        """SLOW_CLASSES 중 지금 **보이는** 것들. 면적은 보지 않는다."""
        return [c for c in cfg.SLOW_CLASSES if c in areas]

    def _car_slow(self, areas):
        """앞의 모형 차량이 CAR_SLOW_AREA 를 넘었으면 (클래스, 면적). 없으면 None.

        회피(_avoid_side)와 같은 클래스를 보되 임계만 낮다 — 피하기 전에 먼저
        느려지는 단계다. 둘 다 넘으면 면적이 큰 쪽(더 가까운 쪽)을 보고한다.
        """
        hits = [(areas[c][0], c) for c in cfg.AVOID_SIDE
                if c in areas and areas[c][0] >= cfg.CAR_SLOW_AREA]
        if not hits:
            return None
        area, name = max(hits)
        return name, area

    def _avoid_ready(self, side):
        """그 방향 회피를 지금 시작해도 되는지. 쿨다운은 **방향별**이다."""
        return (self.stats['frames'] - self.avoid_done_n[side]
                >= cfg.AVOID_COOLDOWN_FRAMES)

    def _end_avoid(self):
        """진행 중인 회피를 끝내고 **그 방향에** 쿨다운을 건다.

        중간에 반대쪽으로 넘어갈 때도 부른다 — 그러지 않으면 넘어간 쪽 기동이
        끝났을 때 처음 방향이 쿨다운 없이 곧바로 다시 걸린다.
        """
        if self.avoid_side is not None:
            self.avoid_done_n[self.avoid_side] = self.stats['frames']
            self.avoid_side = None

    def _avoid_side(self, areas):
        """회피 방향. 없으면 None.

        AVOID_SIDE 의 클래스 중 DETECTION_AREA_ENTER 임계를 넘은 것을 찾는다.
        둘 다 넘으면 **면적이 큰 쪽**(더 가까운 쪽)을 피한다 — 흰 차를 피해
        오른쪽으로 가는 도중 빨간 차가 가까워지면 그때 왼쪽으로 넘어간다.
        """
        hits = [(areas[c][0], side) for c, side in cfg.AVOID_SIDE.items()
                if c in areas and areas[c][0] >= self.area_enter.get(c, float('inf'))]
        return max(hits)[1] if hits else None

    def _step_avoid(self, ctrl, res):
        """회피 중 한 프레임. 중심선을 옆으로 밀어 Pure Pursuit 을 다시 태운다.

        고정 조향 원호가 아니라 **차선을 계속 따라가며 옆으로 비껴가는** 것이
        요점이다 — 곡선 구간에서도 도로를 벗어나지 않는다. detect.py 가 한쪽
        차선만 잡혔을 때 쓰는 "상수항만 밀기"와 같은 기법이다.

        속도는 AVOID_SPEED 를 쓴다 — 감속(SLOW_SPEED)은 걸지 않는다.
        교차로 직진·회전과 같은 규약이다.

        **복귀는 시간으로만 판정한다.** AVOID_FRAMES 가 지나면 차선을 잡았든
        아니든 차선 추종으로 돌아간다. 예전에는 피하는 쪽 차선이 연속으로
        잡히기를 기다렸는데, 회피 구간에서 차선 인식이 잘 안 돼 그 조건이
        성립하지 않았다.

        기동은 두 단계다. AVOID_FRAMES 동안 피하는 쪽으로 밀고, 이어서
        AVOID_RECOVER_FRAMES 동안 **반대쪽으로 같은 양을 밀어** 옆으로 기운
        차체를 도로와 나란히 되돌린다. 뒤 단계가 짧으므로 각도만 돌아오고
        옆으로 나간 거리는 남는다.
        """
        self.stats['avoid'] += 1
        held = self.stats['frames'] - self.avoid_start_n
        sign = +1.0 if self.avoid_side == 'right' else -1.0
        recovering = held >= cfg.AVOID_FRAMES
        if recovering:
            sign = -sign

        if res.fit_center is not None:
            # BEV x 는 오른쪽으로 증가한다. 상수항만 더하면 평행이동이다.
            shifted = detect.LaneResult(status=res.status,
                                        fit_center=res.fit_center.copy())
            shifted.fit_center[-1] += sign * cfg.AVOID_OFFSET_PX
            out = self.pp(shifted, self.roi_h, self.y_start)
            if out.ok:
                target = out.servo
                if self.invert_servo:
                    target = self.pp.servo(-out.delta_deg)
                self.apply_servo(self.servo_cmd
                                 + self.alpha * (target - self.servo_cmd))
        # 중심선이 없으면 밀 대상이 없다 — 직전 조향을 유지하고 버틴다.

        # 감속 배율은 걸지 않는다 — 교차로 직진·회전과 같은 규약으로, 회피는
        # 자기 속도를 그대로 쓴다. red / right_sign 이 보인다고 회피 중에
        # 속도가 또 절반이 되면 기동이 두 배로 길어진다.
        self.apply_motor(self.avoid_speed)

        if held >= cfg.AVOID_FRAMES + cfg.AVOID_RECOVER_FRAMES:
            print(f'  [회피] {self.avoid_side} 완료 — {held}프레임, '
                  f'쿨다운 {cfg.AVOID_COOLDOWN_FRAMES}프레임')
            self._end_avoid()
            self._set_state('LANE_FOLLOW', f'회피 {held}프레임 완료')
        elif recovering:
            self._set_state(f'AVOID_BACK_{self.avoid_side.upper()}',
                            f'중심선 {sign * cfg.AVOID_OFFSET_PX:+.0f}px, '
                            f'{held - cfg.AVOID_FRAMES}/'
                            f'{cfg.AVOID_RECOVER_FRAMES}프레임')
        else:
            self._set_state(f'AVOID_{self.avoid_side.upper()}',
                            f'중심선 {sign * cfg.AVOID_OFFSET_PX:+.0f}px, '
                            f'{held}/{cfg.AVOID_FRAMES}프레임')

    def _auto_turn_side(self, areas):
        """자동 회전 방향. 없으면 None.

        화살표(left/right)가 표지판보다 우선한다 — 신호는 그 순간의 지시이고,
        표지판은 이미 지나온 구간의 안내일 수 있기 때문이다.

        **셋 다 DETECTION_AREA_ENTER 를 넘어야 한다.** 예전에는 화살표에만
        게이트가 없어서, conf 0.26 짜리 면적 205 단발 오탐 하나에 차가 즉시
        돌아버렸다. images/obstacles 1048장에서 right 의 연속 구간 9개 중
        7개가 1~2프레임짜리 깜빡임이었고(진짜는 60프레임), 그 하나하나가
        회전 명령이었다.
        """
        if self.stats['frames'] - self.turn_done_n < cfg.TURN_COOLDOWN_FRAMES:
            return None
        # left 와 right 가 동시에 잡히면 conf 가 높은 쪽. 동전 던지기를 피한다.
        arrows = [(areas[c][1], side) for c, side in cfg.ARROW_TURN.items()
                  if c in areas
                  and areas[c][0] >= self.area_enter.get(c, float('inf'))]
        if arrows:
            return max(arrows)[1]
        sign = areas.get('right_sign')
        if sign and sign[0] >= self.area_enter.get('right_sign', float('inf')):
            return 'right'
        return None

    def _step_turn(self, res):
        """회전 중 한 프레임. 조향은 고정, 탈출은 차선 재획득으로 판정한다.

        좌회전이면 TURN_BACK_FRAMES 동안 먼저 곧게 후진하고, 회전 시간도
        TURN_LEFT_FRAME_SCALE 배 길다 (서보의 좌/우 가동각이 달라 좌회전
        반경이 크기 때문이다). 후진이 끝나야 회전 시간이 시작된다.

        최소 회전 프레임(_turn_frames) 동안은 탈출 조건을 아예 보지 않는다 — 차선이 아직
        보이는 상태에서 버튼을 누르면 첫 프레임부터 조건이 만족되어 회전이
        0.2초 만에 끝나버리기 때문이다.

        **탈출 판정에 ctrl.ok 를 쓰면 안 된다.** sliding_window 는 한쪽만
        잡히면 LANE_WIDTH_PX 로 반대쪽을 외삽해 status='single' 로 중심선을
        만들고, 그러면 ctrl.ok 가 참이 된다. 회전 도중 긴 가로선이 기울며
        세로에 가까워지는 순간 그걸 한쪽 차선으로 오인해 조기 탈출하고,
        차는 존재하지 않는 중심선을 향하게 된다.
        """
        self.stats['turn'] += 1

        if self.turn_back_left > 0:
            # 후진 단계. 조향은 중립으로 두고 곧게 물러난다.
            self.turn_back_left -= 1
            self._set_state('TURN_LEFT_BACK',
                            f'좌회전 전 후진 {self.turn_back_left}프레임 남음')
            self.apply_servo(cfg.SERVO_CENTER)
            self.apply_motor(-self.turn_speed)
            if self.turn_back_left == 0:
                # 다음 프레임부터가 실제 회전이다. 기준점을 다시 잡아 후진
                # 시간이 TURN_MIN_FRAMES / TURN_TIMEOUT_FRAMES 를 먹지 않게 한다.
                # **sub_state 는 여기서 바꾸지 않는다** — 이번 프레임에 나간
                # 명령은 아직 후진이라, 미리 라벨을 바꾸면 화면과 실제가 어긋난다.
                self.turn_start_n = self.stats['frames']
                print(f'  [회전] 후진 완료 — {self.turn_side} 꺾기 시작')
            return

        # apply_servo 는 평활을 하지 않는다 (EMA 는 차선 추종 분기에서 계산된다).
        # 원호를 늦게 시작하면 반경이 커져 못 돌기 때문에 한 프레임에 넣는다.
        self._set_state(f'TURN_{self.turn_side.upper()}',
                        f'servo {self.turn_servo[self.turn_side]} 고정')
        self.apply_servo(self.turn_servo[self.turn_side])
        self.apply_motor(self.turn_speed)

        min_frames, timeout_frames = self._turn_frames(self.turn_side)
        held = self.stats['frames'] - self.turn_start_n
        if held < min_frames:
            # 최소 회전 구간. 차선이 아직 보이는 상태에서 버튼을 눌렀거나
            # 회전 초반에 가로선을 차선으로 오인해도 여기서 걸러진다.
            return

        solid = (res.status == 'ok' and res.width is not None
                 and abs(res.width - cfg.LANE_WIDTH_PX) < cfg.TURN_WIDTH_TOL_PX)
        self.turn_exit_run = self.turn_exit_run + 1 if solid else 0

        if self.turn_exit_run >= cfg.TURN_EXIT_FRAMES:
            print(f'  [회전] {self.turn_side} 완료 — {held}프레임, 차선 재획득')
            self.turn_side = None
            self.turn_done_n = self.stats['frames']
            self._set_state('LANE_FOLLOW', f'회전 완료 — {held}프레임, 차선 재획득')
        elif held > timeout_frames:
            print(f'  [정지] 회전 {held}프레임 초과 — 차선을 못 잡았습니다')
            self.turn_side = None
            self.turn_done_n = self.stats['frames']
            self.stats['halt'] += 1
            self.stopped = True
            self._set_state('TURN_TIMEOUT',
                            f'{held}프레임 > {timeout_frames} — 차선 못 잡음')
            self.apply_motor(0)

    def _seen(self, classes):
        """지금 보이는 것 중 classes 에 드는 이름들. det 가 없으면 빈 목록."""
        counts = getattr(self.det, 'counts', None) or {}
        return [c for c in classes if counts.get(c, 0) > 0]

    def step(self, frame):
        self.stats['frames'] += 1
        ctrl, roi, res, y_start = self.perceive(frame)

        if self.mode == 'STOP':
            self._set_state('STOP', '웹에서 STOP')
            self.apply_motor(0)
            return ctrl, roi, res, y_start

        if self.mode == 'MANUAL':
            self._set_state('MANUAL', '수동 조종')
            return ctrl, roi, res, y_start

        # 원격 탐지 링크가 끊겼다. 객체 판단 자체를 믿을 수 없으므로
        # 아래 분기를 타기 전에 선다 — "안 보인다"와 "못 받았다"는 다르다.
        if self.link_halt:
            if not self.stopped:
                self.stats['halt'] += 1
                print('  [정지] 원격 탐지 링크 끊김')
            self._set_state('LINK_LOST', '탐지 결과가 너무 묵음')
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
            self._set_state(f'STOP_{stopper[0].upper()}',
                            f'{stopper[0]} 탐지 (거리 게이트 없음)')
            self.stats['object_stop'] += 1
            self.apply_motor(0)
            self.apply_servo(cfg.SERVO_CENTER)
            return ctrl, roi, res, y_start

        # 자동 반응 세 가지가 모두 꺼져 있으면 면적 계산 자체를 건너뛴다.
        areas = (self._areas()
                 if (self.auto_turn or self.red_stop or self.slow_on_sight) else {})

        # 자동 회전은 빨간불 정지보다 **먼저** 본다. 그래야 빨간불에 서 있다가
        # 화살표가 뜨는 순간 회전으로 넘어간다 — 정지 상태에서도 매 프레임
        # 여기를 지나기 때문이다.
        if self.auto_turn:
            side = self._auto_turn_side(areas)
            if side is not None and self.start_turn(side, auto=True):
                self._step_turn(res)
                return ctrl, roi, res, y_start

        # 빨간불이 임계 면적을 넘으면 정지하고, 화살표가 나올 때까지 대기한다.
        if self.red_stop:
            red = areas.get('red')
            if red and red[0] >= self.area_enter.get('red', float('inf')):
                # 면적은 매 프레임 달라지므로 **이름이 아니라 근거에** 넣는다
                self._set_state('STOP_RED',
                                f'면적 {int(red[0])} >= '
                                f'{int(self.area_enter["red"])} — 방향 신호 대기')
                self.stats['object_stop'] += 1
                self.stopped = True
                self.apply_motor(0)
                self.apply_servo(cfg.SERVO_CENTER)
                return ctrl, roi, res, y_start

        # 정적 장애물 회피. 차선 추종의 변형이므로 바로 앞에 둔다 — 사람·
        # 화살표·빨간불(3~5번)은 여전히 회피보다 우선한다.
        if self.avoid:
            side = self._avoid_side(areas)
            if (side is not None and side != self.avoid_side
                    and self._avoid_ready(side)):
                self._end_avoid()   # 반대쪽으로 넘어가는 경우 — 이쪽도 쿨다운
                self.avoid_side = side
                self.avoid_start_n = self.stats['frames']
                print(f'  [회피] {side} 시작 — 중심선 {cfg.AVOID_OFFSET_PX}px 이동')
            if self.avoid_side is not None:
                self._step_avoid(ctrl, res)
                return ctrl, roi, res, y_start

        if ctrl.ok:
            # 감속은 **차선 추종에만** 건다. 교차로 직진과 회전은 자기 속도가
            # 있다. 값은 배율이 아니라 **절대 속도**이고, 조건이 여러 개 걸리면
            # 차례로 더 느린 값만 취한다 — 감속 조건이 속도를 올리지는 않는다.
            speed, state, reason = self.speed, 'LANE_FOLLOW', '차선 추종'
            seen = self._slow_seen(areas) if self.slow_on_sight else []
            car = self._car_slow(areas) if self.slow_on_sight else None
            if seen and cfg.SLOW_SPEED < speed:
                speed = cfg.SLOW_SPEED
                state = 'LANE_FOLLOW_SLOW'
                reason = f'{", ".join(seen)} 보임 — 속도 {speed}'
            if car is not None and cfg.CAR_SLOW_SPEED < speed:
                speed = cfg.CAR_SLOW_SPEED
                state = 'LANE_FOLLOW_SLOW'
                reason = (f'{car[0]} 면적 {int(car[1])} >= '
                          f'{cfg.CAR_SLOW_AREA} — 속도 {speed}')
            self._set_state(state, reason)
            self.stats['lane_follow'] += 1
            self.stats['ok'] += 1
            self.fail_streak = 0
            self.crossroad_frames = 0
            self.stopped = False
            target = ctrl.servo
            if self.invert_servo:
                # **서보값을 반사하면 안 된다.** 좌우 단위당 각도가 달라
                # (SERVO_LEFT_RATIO) 2*90-servo 는 최대각에서만 맞고 중간
                # 각도에서 최대 7도까지 어긋난다. 뒤집을 것은 **조향각**이고,
                # 매핑은 그 뒤에 한 번만 태운다.
                target = self.pp.servo(-ctrl.delta_deg)
            self.apply_servo(self.servo_cmd + self.alpha * (target - self.servo_cmd))
            self.apply_motor(speed)
            return ctrl, roi, res, y_start

        # 차선이 없는데 대상 객체도 안 보인다 -> 교차로로 보고 직진한다
        if not self._seen(self.target_classes):
            self.crossroad_frames += 1
            self.fail_streak = 0
            if self.crossroad_frames > cfg.CROSSROAD_MAX_FRAMES:
                # 이만큼 직진했는데도 차선이 안 잡히면 교차로가 아니었던 것이다
                self._set_state('CROSSROAD_TIMEOUT',
                                f'{self.crossroad_frames}프레임 > '
                                f'{cfg.CROSSROAD_MAX_FRAMES} — 교차로가 아니었음')
                if not self.stopped:
                    self.stats['halt'] += 1
                    print(f'  [정지] 교차로 직진 {self.crossroad_frames}프레임 초과')
                self.stopped = True
                self.apply_motor(0)
                return ctrl, roi, res, y_start

            self._set_state('CROSSROAD_STRAIGHT', '차선 없음 + 대상 객체 없음')
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
            self._set_state('FAIL_SAFE',
                            f'연속 {self.fail_streak}프레임 검출 실패')
            if not self.stopped:
                self.stats['halt'] += 1
                print(f'  [정지] 연속 {self.fail_streak}프레임 검출 실패')
            self.stopped = True
            self.apply_motor(0)
        else:
            # 프레임 수는 이름이 아니라 근거에 — 매 프레임 새 전이가 되지 않게
            self._set_state('HOLD', f'검출 실패 {self.fail_streak}/{self.max_fail}, '
                                    '직전 조향 유지')
            self.apply_servo(self.servo_cmd)
            self.apply_motor(self.speed)

        return ctrl, roi, res, y_start
