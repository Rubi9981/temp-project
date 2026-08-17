"""Pure Pursuit 기하 단위 테스트.

오프라인에서 **참/거짓이 확정되는 유일한 검증 수단**이다. 74장의 정적 이미지로는
조향 명령을 계산할 수 있을 뿐, 그 명령이 차를 차선 안에 유지하는지는 알 수 없다.
그래서 기하만이라도 확실히 못 박는다.

이 프로젝트에 테스트 인프라도 pytest 의존성도 없으므로 plain assert 로 쓴다.

    python3 test_control.py
"""
import math

import numpy as np

import _path  # noqa: F401  — 상위 폴더를 import 경로에 추가 (아래 형제 모듈용)
import bev as bevlib
import config as cfg
import control
import detect

M = cfg.default_metric()        # 실측값 유무와 무관하게 고정값으로 테스트
PP = control.PurePursuit(metric=M)

passed = failed = 0


def check(name, cond, detail=''):
    global passed, failed
    if cond:
        passed += 1
        print(f'  PASS  {name}')
    else:
        failed += 1
        print(f'  FAIL  {name}   {detail}')


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ==============================================================================
print('\n[1] 좌표 변환')
# ==============================================================================
# 차량 중심선 바로 위, BEV 최하단 -> Y=0, X=뒤축 오프셋
X, Y = bevlib.bev_to_vehicle(M.vehicle_center_x_px, cfg.H - 1, M)
check('중심선 최하단 -> Y=0', close(float(Y), 0.0))
check('중심선 최하단 -> X=rear_axle_offset', close(float(X), M.rear_axle_offset_cm))

# 왼쪽(작은 x)이 Y 양수
X2, Y2 = bevlib.bev_to_vehicle(M.vehicle_center_x_px - 100, cfg.H - 1, M)
check('왼쪽이 Y +', float(Y2) > 0, f'Y={float(Y2):.2f}')

# 왕복 변환
for xp, yp in [(320, 479), (100, 300), (600, 264)]:
    Xa, Ya = bevlib.bev_to_vehicle(xp, yp, M)
    xb, yb = bevlib.vehicle_to_bev(Xa, Ya, M)
    check(f'왕복 변환 ({xp},{yp})', close(float(xb), xp, 1e-6) and close(float(yb), yp, 1e-6),
          f'-> ({float(xb):.3f},{float(yb):.3f})')


# ==============================================================================
print('\n[2] 곡률 공식')
# ==============================================================================
check('정면 목표 -> kappa=0', close(PP.curvature(50.0, 0.0), 0.0))

# kappa = 2Y/Ld^2 가 2 sin(alpha)/Ld 와 같은지 (같은 식의 두 표현)
for Xg, Yg in [(50.0, 10.0), (30.0, -8.0), (40.0, 25.0)]:
    ld = math.hypot(Xg, Yg)
    alpha = math.atan2(Yg, Xg)
    check(f'kappa 두 표현 일치 ({Xg},{Yg})',
          close(PP.curvature(Xg, Yg), 2 * math.sin(alpha) / ld, 1e-12))

# 좌우 대칭
check('좌우 부호 대칭',
      close(PP.curvature(50.0, 12.0), -PP.curvature(50.0, -12.0)))
check('왼쪽 목표 -> kappa +', PP.curvature(50.0, 12.0) > 0)


# ==============================================================================
print('\n[3] 원호가 실제로 목표점을 지나는가 (자기무결성)')
# ==============================================================================
# Pure Pursuit 의 정의: 원점에서 진행방향에 접하는 반경 1/kappa 원이 목표점을 지난다.
# curvature() 가 옳다면 이 성질이 성립해야 한다.
for Xg, Yg in [(50.0, 10.0), (30.0, -8.0), (40.0, 25.0), (60.0, 2.0), (20.0, -15.0)]:
    kappa = PP.curvature(Xg, Yg)
    ld = math.hypot(Xg, Yg)
    ax, ay = control.arc_points(kappa, ld * 2.0, n=20000)
    dmin = float(np.min(np.hypot(ax - Xg, ay - Yg)))
    check(f'원호가 목표점 통과 ({Xg},{Yg})', dmin < 0.5, f'최소거리 {dmin:.4f}cm')

# 직선 특수 케이스
ax, ay = control.arc_points(0.0, 50.0, n=100)
check('kappa=0 이면 직선', float(np.max(np.abs(ay))) < 1e-9)


# ==============================================================================
print('\n[4] 조향각 / 서보')
# ==============================================================================
check('kappa=0 -> delta=0', close(PP.steer_angle(0.0), 0.0))
check('delta=0 -> servo=중립', PP.servo(0.0) == cfg.SERVO_CENTER)

# 자전거 모델 손계산
kappa = 0.01
check('delta = atan(L*kappa)',
      close(PP.steer_angle(kappa), math.degrees(math.atan(M.wheelbase_cm * kappa)), 1e-9))

# 최대 조향각 클립 -> 서보 한계
big_left = PP.steer_angle(PP.curvature(10.0, 30.0))
check('과한 좌선회는 최대각으로 클립', close(big_left, cfg.MAX_STEER_DEG, 1e-9),
      f'delta={big_left:.3f}')
check('최대 좌선회 -> servo=SERVO_MIN', PP.servo(big_left) == cfg.SERVO_MIN,
      f'servo={PP.servo(big_left)}')
check('최대 우선회 -> servo=SERVO_MAX', PP.servo(-big_left) == cfg.SERVO_MAX,
      f'servo={PP.servo(-big_left)}')
check('좌선회는 서보값이 작아진다 (L_5_Capture 규약)', PP.servo(10.0) < cfg.SERVO_CENTER)
check('서보는 항상 범위 안',
      all(cfg.SERVO_MIN <= PP.servo(d) <= cfg.SERVO_MAX for d in range(-180, 181, 5)))


# ==============================================================================
print('\n[5] 목표점 탐색')
# ==============================================================================
ROI_H, Y_START = cfg.H - int(cfg.H * cfg.ROI_Y_RATIO), int(cfg.H * cfg.ROI_Y_RATIO)

# 중심선이 차량 정중앙을 따라 곧게 뻗은 경우
straight = np.array([0.0, 0.0, M.vehicle_center_x_px])
goal_bev, goal_cm, clamped = PP.lookahead_point(straight, ROI_H, Y_START)
check('직선 중심선 -> Y=0', close(goal_cm[1], 0.0, 1e-9))
check('직선 중심선 -> 목표거리 = L_d', close(math.hypot(*goal_cm), PP.lookahead_cm, 0.2),
      f'{math.hypot(*goal_cm):.3f}')
check('직선 중심선 -> 클램프 없음', not clamped)

res = detect.LaneResult(status='ok', fit_center=straight, eval_y=ROI_H / 2)
out = PP(res, ROI_H, Y_START)
check('직선 -> servo 중립 (+-3)', abs(out.servo - cfg.SERVO_CENTER) <= 3, f'servo={out.servo}')
check('직선 -> ok', out.ok)

# L_d 가 시야 밖 -> 클램프하고 예외 없이 동작
far = control.PurePursuit(metric=M, lookahead_cm=500.0)
gb, gc, clamped = far.lookahead_point(straight, ROI_H, Y_START)
check('L_d 시야 밖 -> 클램프', clamped)
check('클램프 시 ROI 최상단 사용', close(gb[1], Y_START, 1.0), f'y={gb[1]:.1f}')
check('클램프해도 ok', far(res, ROI_H, Y_START).ok)

# ROI 좌표는 y 가 **아래(차 쪽)로 증가**한다. 따라서 1차 계수가 음수면
# 전방(작은 y)의 중심선 x 가 커지는 게 아니라 작아진다 = 목표점이 왼쪽.
# 부호를 헷갈리기 쉬운 지점이라 목표점의 Y 부호로 의도를 못 박고 간다.
left_curve = np.array([0.0, -0.7, M.vehicle_center_x_px])
_, goal_l, _ = PP.lookahead_point(left_curve, ROI_H, Y_START)
check('left_curve 의 목표점이 실제로 왼쪽(Y>0)', goal_l[1] > 0, f'Y={goal_l[1]:.2f}')
out_l = PP(detect.LaneResult(status='ok', fit_center=left_curve), ROI_H, Y_START)
check('좌향 곡선 -> 좌선회 (servo < 90)', out_l.servo < cfg.SERVO_CENTER,
      f'servo={out_l.servo} delta={out_l.delta_deg:+.2f}')

right_curve = np.array([0.0, 0.7, M.vehicle_center_x_px])
_, goal_r, _ = PP.lookahead_point(right_curve, ROI_H, Y_START)
check('right_curve 의 목표점이 실제로 오른쪽(Y<0)', goal_r[1] < 0, f'Y={goal_r[1]:.2f}')
out_r = PP(detect.LaneResult(status='ok', fit_center=right_curve), ROI_H, Y_START)
check('우향 곡선 -> 우선회 (servo > 90)', out_r.servo > cfg.SERVO_CENTER,
      f'servo={out_r.servo} delta={out_r.delta_deg:+.2f}')
check('좌우 곡선 대칭', out_l.servo + out_r.servo == 2 * cfg.SERVO_CENTER,
      f'{out_l.servo} + {out_r.servo}')

# 중심선이 없으면 ok=False
check('중심선 없으면 ok=False',
      not PP(detect.LaneResult(status='fail'), ROI_H, Y_START).ok)


# ==============================================================================
print(f'\n{"=" * 50}')
print(f'  통과 {passed} / 실패 {failed}')
print('=' * 50)
raise SystemExit(1 if failed else 0)
