"""BEV 픽셀 <-> cm 환산 캘리브레이션.

Pure Pursuit 은 "전방 L_d cm 지점"을 다루므로 BEV가 미터법이어야 한다.
현재 BEV는 dst 사각형을 640x480 전체로 잡아 만든 것이라 x축과 y축 스케일이
서로 다르고 둘 다 미측정 상태다. 이 도구가 그걸 채운다.

준비:
  1. 바닥에 **뒤축 기준** 두 지점을 테이프로 표시 (예: 30cm, 60cm)
  2. 차를 그 자리에 두고 캡처
  3. 차선 실폭(띠 중심 <-> 띠 중심)과 휠베이스를 자로 잰다

기준점을 **2개** 쓰는 게 요점이다. 1개로는 스케일만 나오고 뒤축 오프셋
(BEV 최하단 행이 뒤축에서 몇 cm 앞인지)을 구할 수 없다.

x축 스케일은 새로 잴 필요가 없다 — gt.json 라벨 51쌍의 BEV 차선 폭 평균
457.5px 을 실폭으로 나누면 된다.

    python calibrate_metric.py --image ../project/captures/ruler.jpg \\
        --lane-width-cm 40 --wheelbase-cm 16 --near-cm 30 --far-cm 60

조작:
    좌클릭 1  : 가까운 기준점 (--near-cm)
    좌클릭 2  : 먼 기준점     (--far-cm)
    좌클릭 3  : 차량 중심선   (생략하면 화면 중앙 320)
    u : 취소   s : 저장 후 종료   q : 저장 없이 종료
"""
import argparse
import os

import cv2

import bev as bevlib
import config as cfg

clicks = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 3:
        clicks.append((x, y))
        print(f'  클릭 {len(clicks)}: ({x}, {y})')


def solve(near_px_y, far_px_y, near_cm, far_cm, lane_width_cm, wheelbase_cm,
          center_x_px):
    """클릭 결과를 Metric 으로 푼다.

    X = rear_axle_offset + (H-1 - y) / px_per_cm_y  를 두 점에 대해 세우고 푼다.
    BEV는 아래로 갈수록 가까우므로 near 의 y 가 far 보다 크다.
    """
    dy = near_px_y - far_px_y
    dcm = far_cm - near_cm
    if dy <= 0 or dcm <= 0:
        raise SystemExit('먼 기준점이 가까운 기준점보다 화면 아래에 있습니다. '
                         '클릭 순서(가까운 점 -> 먼 점)를 확인하세요.')

    px_per_cm_y = dy / dcm
    rear_axle_offset_cm = near_cm - (cfg.H - 1 - near_px_y) / px_per_cm_y

    return cfg.Metric(
        px_per_cm_x=cfg.GT_LANE_WIDTH_PX / lane_width_cm,
        px_per_cm_y=px_per_cm_y,
        vehicle_center_x_px=center_x_px,
        rear_axle_offset_cm=rear_axle_offset_cm,
        wheelbase_cm=wheelbase_cm,
        lane_width_cm=lane_width_cm,
        measured=True,
    )


def report(m, near_cm, far_cm):
    print()
    print('=' * 58)
    print(f'  px_per_cm_x         = {m.px_per_cm_x:8.3f}   '
          f'(= {cfg.GT_LANE_WIDTH_PX} / {m.lane_width_cm})')
    print(f'  px_per_cm_y         = {m.px_per_cm_y:8.3f}')
    print(f'  가로:세로 스케일 비 = {m.px_per_cm_x / m.px_per_cm_y:8.3f}   '
          f'(1.0 이 아니면 BEV가 비등방)')
    print(f'  vehicle_center_x_px = {m.vehicle_center_x_px:8.1f}')
    print(f'  rear_axle_offset_cm = {m.rear_axle_offset_cm:8.2f}   '
          f'(뒤축 -> BEV 최하단이 비추는 지면)')
    print(f'  wheelbase_cm        = {m.wheelbase_cm:8.2f}')
    print('-' * 58)
    near_X, _ = bevlib.bev_to_vehicle(m.vehicle_center_x_px, cfg.H - 1, m)
    far_X, _ = bevlib.bev_to_vehicle(m.vehicle_center_x_px,
                                     int(cfg.H * cfg.ROI_Y_RATIO), m)
    print(f'  ROI 가시 범위: 전방 {float(near_X):.1f} ~ {float(far_X):.1f} cm')
    if cfg.LOOKAHEAD_CM > float(far_X):
        print(f'  [경고] LOOKAHEAD_CM={cfg.LOOKAHEAD_CM} 이 가시 범위를 넘습니다. '
              f'매 프레임 클램프됩니다.')
    print('=' * 58)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='기준점이 찍힌 캡처 이미지')
    ap.add_argument('--already-bev', action='store_true', help='이미 BEV인 이미지')
    ap.add_argument('--lane-width-cm', type=float, required=True,
                    help='차선 띠 중심 <-> 띠 중심 실폭')
    ap.add_argument('--wheelbase-cm', type=float, required=True,
                    help='앞축 중심 <-> 뒤축 중심')
    ap.add_argument('--near-cm', type=float, required=True, help='가까운 기준점 거리')
    ap.add_argument('--far-cm', type=float, required=True, help='먼 기준점 거리')
    args = ap.parse_args()

    if not os.path.exists(args.image):
        raise SystemExit(f'이미지가 없습니다: {args.image}')

    frame, _, _ = bevlib.load_bev(args.image, already_bev=args.already_bev)
    if frame is None:
        raise SystemExit(f'이미지를 열 수 없습니다: {args.image}')

    cv2.namedWindow('calibrate')
    cv2.setMouseCallback('calibrate', on_mouse)
    print(__doc__)

    labels = [f'1: NEAR mark ({args.near_cm:.0f}cm)',
              f'2: FAR mark ({args.far_cm:.0f}cm)',
              '3: vehicle centerline (optional)']

    while True:
        view = frame.copy()
        cv2.line(view, (0, cfg.H - 1), (cfg.W - 1, cfg.H - 1), (255, 255, 0), 2)
        y_roi = int(cfg.H * cfg.ROI_Y_RATIO)
        cv2.line(view, (0, y_roi), (cfg.W - 1, y_roi), (255, 255, 0), 1)

        for i, (x, y) in enumerate(clicks):
            color = [(0, 128, 255), (0, 255, 128), (255, 0, 255)][i]
            cv2.circle(view, (x, y), 7, color, -1)
            cv2.line(view, (0, y), (cfg.W - 1, y), color, 1)
            cv2.putText(view, str(i + 1), (x + 10, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        nxt = labels[len(clicks)] if len(clicks) < 3 else 'ready - press s to save'
        cv2.putText(view, f'CLICK {nxt}', (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(view, f'CLICK {nxt}', (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(view, 'u:undo   s:save&quit   q:quit', (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(view, 'u:undo   s:save&quit   q:quit', (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('calibrate', view)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('u'):
            if clicks:
                clicks.pop()
        elif key in (27, ord('q')):
            cv2.destroyAllWindows()
            print('저장하지 않고 종료했습니다.')
            return
        elif key == ord('s'):
            if len(clicks) < 2:
                print('  기준점 2개는 반드시 찍어야 합니다.')
                continue
            cv2.destroyAllWindows()
            center_x = clicks[2][0] if len(clicks) == 3 else cfg.VEHICLE_CENTER_X_PX_DEFAULT
            m = solve(clicks[0][1], clicks[1][1], args.near_cm, args.far_cm,
                      args.lane_width_cm, args.wheelbase_cm, center_x)
            report(m, args.near_cm, args.far_cm)
            path = cfg.save_metric(
                m, note=f'calibrate_metric.py image={os.path.basename(args.image)} '
                        f'near={args.near_cm} far={args.far_cm}')
            print(f'\n저장했습니다: {path}')
            return


if __name__ == '__main__':
    main()
