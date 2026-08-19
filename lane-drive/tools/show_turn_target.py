"""TURN_OFFSET_PX 가 추종점을 얼마나 옮기는지 실제 이미지 위에 그린다.

    python3 tools/show_turn_target.py
    python3 tools/show_turn_target.py --image ../project/captures/1go.jpg --side left
    python3 tools/show_turn_target.py --offsets 0,120,160,200,260 --out /tmp/turn.png

왼쪽은 원본 카메라 이미지(추종점을 역투영해 얹은 것), 오른쪽은 BEV 다.

**추종점은 옆으로만 미는 것이 아니다.** crossroad_driver._turn_servo_table() 은
뒤축에서의 **방사 거리를 L_d 로 유지**한 채 옆으로 민다:

    Y = ±N / px_per_cm_x
    X = sqrt(L_d² - Y²)

그래서 점이 수평으로 미끄러지지 않고 **반지름 L_d 인 원을 따라 돈다** — 옆으로
갈수록 앞쪽 거리 X 가 줄어든다. 그림의 점선 원이 그 궤적이다.

N 이 커지면 조향각이 MAX_STEER_DEG 에서 포화하므로, 그 이상은 점만 움직이고
서보 명령은 그대로다. 범례의 servo 값이 같아지는 지점이 그 경계다.
"""
import argparse
import glob
import math
import os

import cv2
import numpy as np

import _path  # noqa: F401
import bev as bevlib
import binarize
import config as cfg
import control
import detect

# 오프셋별 색 (BGR). 작은 값 -> 큰 값 순으로 노랑에서 빨강으로 간다.
_COLORS = [(0, 255, 255), (0, 200, 255), (0, 150, 255), (0, 100, 255),
           (0, 60, 255), (0, 0, 255), (60, 0, 220)]


def goal_for_offset(offset_px, side, metric):
    """오프셋 -> (차량좌표 X, Y, kappa, delta, servo, 반경).

    crossroad_driver._turn_servo_table() 과 같은 식을 쓴다.
    """
    pp = control.PurePursuit(metric=metric)
    sign = +1.0 if side == 'left' else -1.0
    Y = sign * offset_px / metric.px_per_cm_x
    X = math.sqrt(max(cfg.LOOKAHEAD_CM ** 2 - Y * Y, 1e-9))
    kappa = pp.curvature(X, Y)
    delta = pp.steer_angle(kappa)
    servo = pp.servo(delta)
    # 실제로 그려지는 원호의 반경은 요청값이 아니라 **클립된 조향각** 기준이다.
    # 포화 구간에서 이 둘이 갈라진다 — 그래서 delta 로 되돌려 계산한다.
    radius = (metric.wheelbase_cm / math.tan(math.radians(abs(delta)))
              if abs(delta) > 1e-6 else float('inf'))
    return X, Y, kappa, delta, servo, radius


def text(img, s, org, color=(255, 255, 255), scale=0.5):
    """검은 테두리를 깔아 어떤 배경에서도 읽히게 그린다."""
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
                cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(
        description='TURN_OFFSET_PX 가 추종점을 얼마나 옮기는지 그린다')
    ap.add_argument('--image', metavar='PATH',
                    help='생략하면 project/captures/ 에서 하나 고른다')
    ap.add_argument('--side', default='right', choices=('left', 'right'))
    ap.add_argument('--offsets', default='0,120,160,200,260',
                    help='쉼표로 구분한 BEV 픽셀 오프셋')
    ap.add_argument('--binarize', default='adaptive', choices=list(binarize.BACKENDS))
    ap.add_argument('--out', default='turn_target.png')
    ap.add_argument('--show', action='store_true', help='창으로도 띄운다')
    args = ap.parse_args()

    path = args.image
    if path is None:
        cand = sorted(glob.glob(os.path.join(cfg.CAPTURES_DIR, '*.jpg')))
        if not cand:
            raise SystemExit(f'이미지가 없습니다: {cfg.CAPTURES_DIR}. --image 로 지정하세요.')
        path = cand[0]

    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f'이미지를 읽을 수 없습니다: {path}')
    img = cv2.resize(img, (cfg.W, cfg.H))

    metric = cfg.get_metric()
    offsets = [int(v) for v in args.offsets.split(',') if v.strip()]

    bev, matrix = bevlib.warp_image(img)
    inv = np.linalg.inv(matrix)
    canvas = bev.copy()
    orig = img.copy()

    # --- 배경: ROI 경계와 차량 중심선 ---
    y_start = int(cfg.H * cfg.ROI_Y_RATIO)
    cv2.line(canvas, (0, y_start), (cfg.W, y_start), (90, 90, 90), 1)
    text(canvas, f'ROI y={y_start}', (8, y_start - 6), (150, 150, 150), 0.45)
    cx = int(metric.vehicle_center_x_px)
    cv2.line(canvas, (cx, y_start), (cx, cfg.H), (90, 90, 90), 1)

    # --- 추종점이 놓이는 원 (반지름 L_d, 뒤축 중심) ---
    th = np.linspace(-math.pi / 2, math.pi / 2, 200)
    cxs, cys = bevlib.vehicle_to_bev(cfg.LOOKAHEAD_CM * np.cos(th),
                                     cfg.LOOKAHEAD_CM * np.sin(th), metric)
    pts = np.stack([cxs, cys], 1).astype(np.int32)
    for i in range(0, len(pts) - 1, 6):            # 점선
        cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), (180, 180, 180), 2)
    text(canvas, f'L_d={cfg.LOOKAHEAD_CM:.0f}cm', (int(cxs[0]) + 6, int(cys[0])),
         (180, 180, 180), 0.45)

    # --- 차선 추종이 내는 정상 추종점 (비교용) ---
    roi, roi_y = bevlib.roi_of(bev)
    res = detect.sliding_window(binarize.BACKENDS[args.binarize](roi))
    ctrl = control.PurePursuit(metric=metric)(res, roi.shape[0], roi_y)
    if ctrl.ok:
        gx, gy = ctrl.goal_bev
        cv2.circle(canvas, (int(gx), int(gy)), 9, (0, 0, 0), -1)
        cv2.circle(canvas, (int(gx), int(gy)), 7, (255, 0, 255), -1)
        text(canvas, 'lane', (int(gx) - 14, int(gy) - 14), (255, 0, 255), 0.45)

    # --- 오프셋별 추종점과 그 결과 원호 ---
    print(f'  이미지 {os.path.basename(path)}   방향 {args.side}   '
          f'L_d={cfg.LOOKAHEAD_CM}cm\n')
    print(f'{"N(px)":>6} {"옆으로":>8} {"BEV x":>7} {"BEV y":>7} {"dx":>6} {"dy":>6} '
          f'{"delta":>7} {"반경":>8} {"servo":>6}')
    base_x = base_y = base_px = base_py = None
    rows = []
    for i, N in enumerate(offsets):
        color = _COLORS[i % len(_COLORS)]
        X, Y, kappa, delta, servo, radius = goal_for_offset(N, args.side, metric)
        bx, by = bevlib.vehicle_to_bev(X, Y, metric)
        if base_x is None:
            base_x, base_y = bx, by

        # 결과 원호 — 이 조향을 유지했을 때 실제로 그려지는 궤적
        aX, aY = control.arc_points(kappa, 45.0, 60)
        ax, ay = bevlib.vehicle_to_bev(aX, aY, metric)
        arc = np.stack([ax, ay], 1).astype(np.int32)
        cv2.polylines(canvas, [arc], False, color, 2, cv2.LINE_AA)

        cv2.circle(canvas, (int(bx), int(by)), 9, (0, 0, 0), -1)
        cv2.circle(canvas, (int(bx), int(by)), 7, color, -1)
        text(canvas, f'{N}', (int(bx) - 10, int(by) + 26), color, 0.5)

        # 원본 이미지에 역투영
        px, py = bevlib.project_points([[bx, by]], inv)[0]
        if i == 0:
            base_px, base_py = px, py       # N=0 (직진 추종점) 이 기준
        else:
            cv2.line(orig, (int(base_px), int(base_py)), (int(px), int(py)),
                     color, 1, cv2.LINE_AA)
        cv2.circle(orig, (int(px), int(py)), 8, (0, 0, 0), -1)
        cv2.circle(orig, (int(px), int(py)), 6, color, -1)
        text(orig, f'{N}', (int(px) + 9, int(py) + 4), color, 0.45)

        rows.append((N, color, delta, radius, servo, Y))
        r = '   max' if math.isinf(radius) else f'{radius:7.1f}'
        print(f'{N:>6} {abs(Y):>7.2f}cm {bx:>7.1f} {by:>7.1f} '
              f'{bx-base_x:>+6.0f} {by-base_y:>+6.0f} '
              f'{delta:>+6.1f}도 {r}cm {servo:>6}')

    # --- 범례 ---
    for i, (N, color, delta, radius, servo, Y) in enumerate(rows):
        y = 24 + i * 20
        cv2.circle(canvas, (16, y - 4), 6, color, -1)
        r = 'max' if math.isinf(radius) else f'{radius:.1f}cm'
        text(canvas, f'N={N:<4d} {abs(Y):4.1f}cm  servo {servo:<4d} '
                     f'{delta:+5.1f}deg  R={r}', (30, y), color, 0.45)

    text(canvas, 'BEV', (cfg.W - 60, cfg.H - 12), (255, 255, 255), 0.6)
    text(orig, f'ORIGINAL  {os.path.basename(path)}', (10, cfg.H - 12),
         (255, 255, 255), 0.5)

    out = np.hstack([orig, canvas])
    cv2.imwrite(args.out, out)
    print(f'\n  저장: {args.out}')
    if args.show:
        cv2.imshow('turn target', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
