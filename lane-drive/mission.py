"""미션 상태 관리자 — 탐지한 객체 종류에 따라 주행 상태를 관리한다.

**아직 주행에 연결되어 있지 않다.** `update()` 가 `Intent` 를 내지만 아무도
소비하지 않는다. `loop.py` 는 상태를 화면에 띄우기만 하고, 차를 실제로 움직이는
것은 여전히 `driver.py` / `crossroad_driver.py` 다.

    mission = MissionManager()
    mission.observe(n, ctrl, det)       # loop.py 가 매 프레임 부른다
    mission.status_str                  # 'RIGHT_SIGN — right_sign 면적 3120 >= 2500'
    mission.history                     # [(프레임, 이전, 이후, 사유), ...]

세 조각으로 나뉜다.

    Observation      한 프레임의 판단 입력. ctrl/det 를 여기서 한 번만 읽는다
    Intent           "이렇게 하고 싶다". 지금은 아무도 안 쓴다
    MissionManager   상태 하나를 들고 전이를 결정한다

**왜 상태 기계인가.** 표지판·신호등은 한 프레임의 탐지로 판단할 수 없다.
표지판은 다가가면 화면 밖으로 나가버리므로(관측된 연속 구간 16~30프레임, 박스
y2 165~279) 매 프레임 `if 표지판이 보이면` 으로 짜면 사라지는 순간 조건이 거짓이
되어 판단이 풀린다. 봤다는 사실을 **래치**해야 한다.

진입은 **bbox 면적**으로 한다 — 면적이 곧 거리 대용이라, 트랙 반대편에 작게
잡힌 표지판에 반응하지 않게 된다.

상태는 **무엇을 봤는가**로 나뉘고, 각 상태는 자기가 반응하는 **종류(kinds)**를
가진다. 한 종류라도 면적 임계를 넘으면 그 상태로 들어간다.

    PERSON          human
    CAR             car_white, car_red
    TRAFFIC_LIGHT   red, left, right
    RIGHT_SIGN      right_sign

복귀 규칙은 두 종류뿐이다 (`_StateDef.exit_rule`).

    STRAIGHT_N   직선 차선이 N프레임 연속 잡히면 복귀 (기본)
    GONE         진입시킨 종류가 N프레임 연속 안 보이면 복귀

`CAR` 만 GONE 을 쓴다. 모형 차량은 **회피 대상**이라 시야에 있는 동안이 곧 회피
구간이고, 지나쳐 안 보이면 끝난 것이다. 2차선 변경 중에는 차선이 직선이 아니므로
STRAIGHT_N 으로는 영영 복귀하지 못한다. GONE 은 **진입시킨 그 종류**를 추적하므로
흰 차량으로 들어갔으면 흰 차량이 사라질 때까지 유지된다.

**방향(좌/우)은 아직 가르지 않는다.** `TRAFFIC_LIGHT` 는 red / left / right 를
한 상태로 받는다. 실제로 좌회전이냐 우회전이냐를 읽어 기동을 시작하는 것은
아직 이 모듈의 일이 아니다 (`crossroad_driver.start_turn()` 이 웹 버튼으로 받는다).
"""
import collections
from dataclasses import dataclass, field

import config as cfg

# -----------------------------
# 상태 이름 (crossroad_driver 의 sub_state 와 같이 평범한 문자열을 쓴다)
# -----------------------------
LANE_FOLLOW = 'LANE_FOLLOW'
PERSON = 'PERSON'
CAR = 'CAR'
TRAFFIC_LIGHT = 'TRAFFIC_LIGHT'
RIGHT_SIGN = 'RIGHT_SIGN'
CROSSING = 'CROSSING'
HALT = 'HALT'

# 복귀 규칙
STRAIGHT_N = 'STRAIGHT_N'
GONE = 'GONE'


@dataclass(frozen=True)
class _StateDef:
    kinds: tuple            # 이 상태가 반응하는 종류. 면적 임계를 넘으면 진입한다
    action: str             # 이 상태에서 내는 Intent.action
    exit_rule: str


STATES = {
    LANE_FOLLOW:   _StateDef((), 'FOLLOW', STRAIGHT_N),
    PERSON:        _StateDef(('human',), 'HOLD', STRAIGHT_N),
    CAR:           _StateDef(('car_white', 'car_red'), 'HOLD', GONE),
    TRAFFIC_LIGHT: _StateDef(('red', 'left', 'right'), 'HOLD', STRAIGHT_N),
    RIGHT_SIGN:    _StateDef(('right_sign',), 'FOLLOW', STRAIGHT_N),
    CROSSING:      _StateDef((), 'STRAIGHT', STRAIGHT_N),
    HALT:          _StateDef((), 'STOP', STRAIGHT_N),
}

# 종류로 진입하는 상태를 볼 순서. **안전이 먼저다** — 사람과 차량을 신호등보다
# 앞에 둔다. 여러 종류가 동시에 임계를 넘으면 이 순서로 하나가 정해진다.
ENTER_ORDER = (PERSON, CAR, TRAFFIC_LIGHT, RIGHT_SIGN)


# ==============================================================================
# 입력
# ==============================================================================
@dataclass(frozen=True)
class Observation:
    """한 프레임의 미션 판단 입력.

    MissionManager 는 ctrl/res/det 를 직접 보지 않고 이것만 본다. 그래야 합성
    입력을 그대로 먹여 전이를 확인할 수 있다 (하드웨어·카메라 불필요).
    """
    n: int = 0                          # 프레임 번호. 타임아웃 계산에 쓴다
    lane_ok: bool = False
    kappa: float = 0.0                  # 곡률 1/cm. + 가 좌선회
    straight: bool = False              # lane_ok 이고 |kappa| 가 임계 미만
    objects: dict = field(default_factory=dict)   # {종류: (면적, conf)}
    objects_fresh: bool = True          # 아래 주석 참조
    link_ok: bool = True
    age_ms: float = 0.0

    @classmethod
    def from_frame(cls, n, ctrl, det=None):
        """주행 루프의 값에서 만든다. **탐지 안정화를 끼울 자리가 여기다.**"""
        straight_limit = cfg.MISSION_STRAIGHT_KAPPA
        lane_ok = bool(ctrl.ok)
        kappa = float(ctrl.kappa)

        if det is None:
            return cls(n=n, lane_ok=lane_ok, kappa=kappa,
                       straight=lane_ok and abs(kappa) < straight_limit,
                       objects={}, objects_fresh=False)

        link = getattr(det, 'link', None) or {}
        age_ms = float(link.get('age_ms', 0.0))
        link_ok = bool(link.get('ok', True))

        # 종류별로 **가장 큰 박스** 하나만 남긴다. 면적이 거리 대용이므로
        # 같은 종류가 여러 개면 가장 가까운 것이 판단 기준이 된다.
        objects = {}
        for x1, y1, x2, y2, name, conf in (det.boxes or []):
            area = float((x2 - x1) * (y2 - y1))
            if area > objects.get(name, (0.0, 0.0))[0]:
                objects[name] = (area, float(conf))

        return cls(
            n=n, lane_ok=lane_ok, kappa=kappa,
            straight=lane_ok and abs(kappa) < straight_limit,
            objects=objects,
            # **"안 보인다"와 "결과를 못 받았다"는 다르다.** 결과가 묵었으면
            # objects 를 빈 것으로 해석하면 안 된다 — 그러면 GONE 복귀가 거짓으로
            # 발동해 회피 도중에 상태가 풀린다. 진입도 하지 않고 GONE 카운터도
            # 세지 않도록 fresh=False 로 알린다.
            objects_fresh=age_ms <= cfg.MISSION_MAX_AGE_MS,
            link_ok=link_ok,
            age_ms=age_ms,
        )


# ==============================================================================
# 출력
# ==============================================================================
@dataclass(frozen=True)
class Intent:
    """상태 기계가 "이렇게 하고 싶다"고 내놓는 것.

    **지금은 아무도 소비하지 않는다.** 주행에 물릴 때 speed/servo 를 채우고
    Driver 가 apply_motor/apply_servo 로 연결하면 된다.
    """
    action: str                 # 'FOLLOW'|'STRAIGHT'|'STOP'|'TURN_LEFT'|'TURN_RIGHT'|'HOLD'
    speed: int = None           # None = 호출자 기본값을 쓰라는 뜻
    servo: int = None           # None = 조향에 관여하지 않겠다는 뜻
    reason: str = ''


# ==============================================================================
# 상태 기계
# ==============================================================================
class MissionManager:
    def __init__(self, area_enter=None, return_frames=None, gone_frames=None,
                 timeout_frames=None):
        self.area_enter = dict(area_enter or cfg.MISSION_AREA_ENTER)
        self.return_frames = (cfg.MISSION_RETURN_FRAMES if return_frames is None
                              else return_frames)
        self.gone_frames = (cfg.MISSION_GONE_FRAMES if gone_frames is None
                            else gone_frames)
        self.timeout_frames = (cfg.MISSION_TIMEOUT_FRAMES if timeout_frames is None
                               else timeout_frames)

        self.state = LANE_FOLLOW
        self.reason = '시작'
        self.trigger = ''           # 이 상태에 들어오게 만든 종류 (GONE 판정용)
        self.entered_n = 0
        self.straight_run = 0       # 연속 직선 프레임
        self.gone_run = 0           # trigger 가 연속으로 안 보인 프레임
        self.history = collections.deque(maxlen=32)

    # -----------------------------
    @property
    def status_str(self):
        """화면 상태표/HUD 에 그대로 나가는 한 줄."""
        return f'{self.state} — {self.reason}'

    def observe(self, n, ctrl, det=None):
        """loop.py 가 부르는 진입점. Observation 을 만들어 update 한다.

        loop.py 가 mission 모듈을 import 하지 않아도 되도록 여기에 둔다.
        """
        return self.update(Observation.from_frame(n, ctrl, det))

    # -----------------------------
    def _go(self, new_state, obs, reason, trigger=''):
        """상태 전이. 같은 상태로의 전이는 무시한다 (이력이 더러워진다)."""
        if new_state == self.state:
            return
        self.history.append((obs.n, self.state, new_state, reason))
        self.state = new_state
        self.reason = reason
        self.trigger = trigger
        self.entered_n = obs.n
        # 새 상태에서 다시 세기 시작한다 — 진입 직전의 직선 프레임이 복귀
        # 조건에 얹혀 즉시 빠져나오는 것을 막는다
        self.straight_run = 0
        self.gone_run = 0

    def _intent(self):
        d = STATES[self.state]
        # speed/servo 는 비워 둔다 — 아직 주행에 연결하지 않는다
        return Intent(action=d.action, reason=self.reason)

    def update(self, obs):
        """한 프레임 처리. Intent 를 돌려주지만 지금은 아무도 쓰지 않는다."""
        self.straight_run = self.straight_run + 1 if obs.straight else 0

        # trigger 가 안 보인 프레임을 센다. **결과가 묵었으면 세지 않는다** —
        # 못 받은 것을 "사라졌다"로 읽으면 회피 도중에 상태가 풀린다.
        if self.trigger and obs.objects_fresh:
            self.gone_run = 0 if self.trigger in obs.objects else self.gone_run + 1

        # 1) 링크가 끊기면 다른 판단을 할 수 없다. 탐지가 안 오는 상태에서
        #    "객체 안 보임"은 아무 뜻도 없기 때문이다.
        if not obs.link_ok:
            self._go(HALT, obs, f'원격 탐지 링크 끊김 (age {obs.age_ms:.0f}ms)')
            return self._intent()

        # 2) 복귀와 타임아웃
        if self.state != LANE_FOLLOW:
            rule = STATES[self.state].exit_rule
            if rule == GONE:
                if self.gone_run >= self.gone_frames:
                    self._go(LANE_FOLLOW, obs,
                             f'{self.trigger} 사라짐 {self.gone_run}프레임')
            elif self.straight_run >= self.return_frames:
                self._go(LANE_FOLLOW, obs, f'직선 차선 {self.straight_run}프레임')

            held = obs.n - self.entered_n
            if (self.state not in (LANE_FOLLOW, HALT)
                    and held >= self.timeout_frames):
                # GONE 상태에도 반드시 건다 — 차량이 영영 시야에 남으면 갇힌다
                self._go(HALT, obs, f'{self.state} 체류 {held}프레임 초과')

        # 3) 진입 (LANE_FOLLOW 에서만)
        if self.state == LANE_FOLLOW:
            if obs.objects_fresh:
                for state in ENTER_ORDER:
                    hit = self._entering(STATES[state].kinds, obs)
                    if hit:
                        name, area = hit
                        self._go(state, obs,
                                 f'{name} 면적 {area:.0f} >= '
                                 f'{self.area_enter[name]:.0f}', trigger=name)
                        break

            # 객체로 설명되지 않는 차선 실종이 교차로다. 객체 진입을 먼저
            # 본 뒤에 확인한다 — crossroad_driver 의 판단 순서와 같다.
            if self.state == LANE_FOLLOW and not obs.lane_ok:
                self._go(CROSSING, obs, '차선 없음')

        return self._intent()

    def _entering(self, kinds, obs):
        """면적 임계를 넘은 첫 종류를 (이름, 면적) 으로. 없으면 None."""
        for name in kinds:
            seen = obs.objects.get(name)
            threshold = self.area_enter.get(name)
            if seen and threshold is not None and seen[0] >= threshold:
                return name, seen[0]
        return None

    def format_history(self):
        """종료 후 "어디서 왜 바뀌었나"를 읽기 위한 것."""
        if not self.history:
            return '  (전이 없음)'
        return '\n'.join(f'  [{n:5d}] {old:15s} -> {new:15s}  {why}'
                         for n, old, new, why in self.history)
