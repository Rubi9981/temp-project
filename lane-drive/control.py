"""Pure Pursuit 조향 제어기.

기존 detect.lane_error() 는 한 지점의 횡방향 오차만 보는 비례 제어라 곡률을
전혀 쓰지 않는다. Pure Pursuit 은 중심선 위의 전방 L_d 지점을 **원호로 통과**
하도록 조향각을 푼다.

    alpha = atan2(Y, X)              목표점 방위각 (뒤축 기준)
    kappa = 2 sin(alpha) / L_d       = 2Y / L_d^2   (Y = L_d sin(alpha) 이므로)
    delta = atan(L * kappa)          자전거 모델 조향각

부호: Y, alpha, kappa, delta 모두 **좌선회가 +**.
서보는 왼쪽이 작은 값(raspi/L_5_Capture.py: ArrowLeft -> 40)이므로
servo = 90 - delta_deg * (서보단위/도) 가 된다. **좌우 계수가 다르다** —
링키지가 비대칭이라 좌선회에는 SERVO_LEFT_RATIO 를 곱한다 (servo() 참조).

프레임 독립 순수 함수로 짰다 — 나중에 Pi 실시간 루프에 그대로 물릴 수 있다.
"""
import math
from dataclasses import dataclass

import numpy as np

import bev as bevlib
import config as cfg


@dataclass
class ControlResult:
    ok: bool
    servo: int = cfg.SERVO_CENTER
    delta_deg: float = 0.0
    kappa: float = 0.0              # 1/cm, + = 좌선회
    goal_bev: tuple = None          # (x_px, y_px) BEV 좌표
    goal_cm: tuple = None           # (X, Y) 차량 좌표
    lookahead_cm: float = None      # 실제 사용된 목표점 거리
    clamped: bool = False           # 요청 L_d 가 시야 밖이라 당겨졌는가
    reason: str = ''                # ok=False 일 때의 사유


class PurePursuit:
    def __init__(self, metric=None, lookahead_cm=None, wheelbase_cm=None,
                 max_steer_deg=None, servo_per_deg=None, left_ratio=None):
        self.m = metric if metric is not None else cfg.get_metric()
        self.lookahead_cm = lookahead_cm or cfg.LOOKAHEAD_CM
        self.wheelbase_cm = wheelbase_cm or self.m.wheelbase_cm
        self.max_steer_deg = max_steer_deg or cfg.MAX_STEER_DEG
        self.servo_per_deg = servo_per_deg or cfg.SERVO_PER_DEG
        # 좌측 링키지 보정. 우측 기준 계수에 이걸 곱해 좌선회에만 쓴다.
        self.left_ratio = left_ratio or cfg.SERVO_LEFT_RATIO

    # -----------------------------
    # 1) 목표점 찾기
    # -----------------------------
    def lookahead_point(self, fit_center, roi_h, y_start):
        """중심선 위에서 뒤축으로부터의 **방사 거리**가 L_d 인 점.

        종방향 거리가 아니라 방사 거리인 것이 정식 정의다. ROI 안에 그런 점이
        없으면 가장 먼 가시점으로 당기고 clamped=True 로 알린다.
        """
        # ROI 를 아래(가까움)에서 위(멂)로 1px 간격 샘플링
        y_roi = np.arange(roi_h - 1, -1, -1, dtype=float)
        x_bev = np.polyval(fit_center, y_roi)
        y_bev = y_roi + y_start

        X, Y = bevlib.bev_to_vehicle(x_bev, y_bev, self.m)
        dist = np.hypot(X, Y)

        hit = np.nonzero(dist >= self.lookahead_cm)[0]
        if hit.size == 0:
            i = int(np.argmax(dist))        # 가장 먼 점으로 클램프
            return (float(x_bev[i]), float(y_bev[i])), (float(X[i]), float(Y[i])), True

        i = int(hit[0])
        if i == 0:
            # 첫 샘플부터 이미 L_d 를 넘음 (목표가 너무 가까움)
            return (float(x_bev[0]), float(y_bev[0])), (float(X[0]), float(Y[0])), False

        # i-1 과 i 사이를 선형 보간해 정확히 L_d 인 점을 만든다
        d0, d1 = dist[i - 1], dist[i]
        t = 0.0 if d1 == d0 else (self.lookahead_cm - d0) / (d1 - d0)
        lerp = lambda a, b: float(a + (b - a) * t)
        return ((lerp(x_bev[i - 1], x_bev[i]), lerp(y_bev[i - 1], y_bev[i])),
                (lerp(X[i - 1], X[i]), lerp(Y[i - 1], Y[i])), False)

    # -----------------------------
    # 2) 기하
    # -----------------------------
    @staticmethod
    def curvature(X_cm, Y_cm):
        """kappa = 2Y / L_d^2. 삼각함수 없이 같은 값이 나온다."""
        ld_sq = X_cm * X_cm + Y_cm * Y_cm
        if ld_sq <= 0:
            return 0.0
        return 2.0 * Y_cm / ld_sq

    def steer_angle(self, kappa):
        """자전거 모델: delta = atan(L * kappa). 최대 조향각으로 클립."""
        delta = math.degrees(math.atan(self.wheelbase_cm * kappa))
        return float(np.clip(delta, -self.max_steer_deg, self.max_steer_deg))

    def servo(self, delta_deg):
        """조향각(deg) -> 서보 명령. 좌선회(delta>0)가 작은 서보값이 된다.

        **좌우 계수가 다르다.** 실측상 같은 28도를 만드는 데 우측은 중립에서
        60단위(servo 150), 좌측은 80단위(servo 10)가 든다. 하나의 계수를 양쪽에
        쓰면 좌선회 28도가 servo 30 으로 나가고, 그건 좌측 60단위라 실제로는
        21도밖에 안 꺾인다 — 좌회전 반경이 우측보다 8cm 커지는 원인이었다.
        """
        per_deg = self.servo_per_deg * (self.left_ratio if delta_deg > 0 else 1.0)
        value = cfg.SERVO_CENTER - delta_deg * per_deg
        return int(round(float(np.clip(value, cfg.SERVO_MIN, cfg.SERVO_MAX))))

    # -----------------------------
    # 3) 한 번에
    # -----------------------------
    def __call__(self, res, roi_h, y_start):
        """detect.LaneResult -> ControlResult."""
        if res.fit_center is None:
            return ControlResult(ok=False, reason='no centerline (detect=sliding 필요)')

        goal_bev, goal_cm, clamped = self.lookahead_point(res.fit_center, roi_h, y_start)
        X, Y = goal_cm
        if X <= 0:
            return ControlResult(ok=False, reason='goal behind rear axle')

        kappa = self.curvature(X, Y)
        delta = self.steer_angle(kappa)
        return ControlResult(
            ok=True,
            servo=self.servo(delta),
            delta_deg=delta,
            kappa=kappa,
            goal_bev=goal_bev,
            goal_cm=goal_cm,
            lookahead_cm=float(math.hypot(X, Y)),
            clamped=clamped,
        )


def arc_points(kappa, length_cm, n=40):
    """조향 결과로 그려질 원호를 차량 좌표 (X, Y) 점열로 만든다.

    반경 R = 1/kappa 인 원이 원점에서 X축에 접한다:
        X = R sin(s/R),  Y = R (1 - cos(s/R))
    kappa -> 0 이면 직선으로 수렴한다.
    """
    s = np.linspace(0.0, length_cm, n)
    if abs(kappa) < 1e-9:
        return s, np.zeros_like(s)
    R = 1.0 / kappa
    theta = s / R
    return R * np.sin(theta), R * (1.0 - np.cos(theta))
