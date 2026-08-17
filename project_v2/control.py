"""Pure Pursuit 기반 기하 조향 제어기."""
import dataclasses
import numpy as np
import config as cfg

@dataclasses.dataclass
class ControlResult:
    ok: bool
    servo: float = float(cfg.SERVO_CENTER)
    delta_deg: float = 0.0
    lookahead_cm: float = cfg.LOOKAHEAD_CM
    goal_bev: tuple = (320.0, 372.0)
    reason: str = ""
    clamped: bool = False

class PurePursuit:
    def __init__(self, metric=None, lookahead_cm=cfg.LOOKAHEAD_CM, servo_per_deg=cfg.SERVO_PER_DEG):
        self.metric = metric or cfg.get_metric()
        self.lookahead_cm = lookahead_cm
        self.servo_per_deg = servo_per_deg

    def __call__(self, res, roi_h, y_start):
        if res is None or getattr(res, 'fit_center', None) is None:
            return ControlResult(ok=False, reason="NO_LANE")

        # BEV 상에서 lookahead 거리만큼 앞선 지점 계산
        lookahead_px = self.lookahead_cm * getattr(self.metric, 'px_per_cm_y', 5.0)
        target_roi_y = max(0.0, float(roi_h) - lookahead_px)
        
        # 중심선 다항식 x = poly(y) 계산
        target_roi_x = float(np.polyval(res.fit_center, target_roi_y))
        
        # BEV 전체 좌표계
        goal_bev_x = target_roi_x
        goal_bev_y = target_roi_y + y_start

        # 차량 중심(x)과의 오차 (cm 단위)
        v_center = getattr(self.metric, 'vehicle_center_x_px', 320.0)
        px_cm_x = getattr(self.metric, 'px_per_cm_x', 5.0)
        dx_px = goal_bev_x - v_center
        dx_cm = dx_px / max(px_cm_x, 1e-6)
        dy_cm = self.lookahead_cm

        # 조향 각도 (deg) 계산
        angle_rad = np.arctan2(dx_cm, dy_cm)
        delta_deg = float(np.degrees(angle_rad))

        # 서보모터 단위로 변환 (90도 중앙 기준)
        servo_target = float(cfg.SERVO_CENTER) + delta_deg * self.servo_per_deg
        clamped = False
        if servo_target < cfg.SERVO_MIN:
            servo_target = float(cfg.SERVO_MIN)
            clamped = True
        elif servo_target > cfg.SERVO_MAX:
            servo_target = float(cfg.SERVO_MAX)
            clamped = True

        return ControlResult(
            ok=True,
            servo=servo_target,
            delta_deg=delta_deg,
            lookahead_cm=self.lookahead_cm,
            goal_bev=(goal_bev_x, goal_bev_y),
            reason="OK",
            clamped=clamped
        )
