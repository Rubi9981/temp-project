"""프레임별 시각 검토 뷰어.

project/lane_detector_v2.py 의 수동 키 조작 루프를 이식하고, 같은 프레임에서
이진화/검출 백엔드를 즉석 전환해 비교할 수 있게 했다.
v2에서 계산만 하고 쓰지 않던 역투영(project_point)도 살렸다.

원본(역투영) + BEV(중심선) + 마스크를 **한 창에 합쳐** 보여주고, d/a 로 폴더의
모든 이미지를 순회한다. 창이 여러 개로 흩어지지 않는다.

gt.json 이 있으면 정답 라벨을 **속 빈 원**으로, 검출 결과를 **채운 원**으로
겹쳐 그린다. 둘이 붙어 있으면 잘 맞은 것이다.

기본 백엔드는 GT 평가 승자인 adaptive + sliding 이라 플래그 없이 바로 쓴다.

    python review.py                       # captures/ 74장 순회
    python review.py --src captures_bev    # captures_bev/ 68장 순회
    python review.py --sort worst          # 오차 큰 순 — 실패 사례부터 (gt.json 필요)
    python review.py --scale 0.6           # 창이 크면 줄이기
    python review.py --export out/         # GUI 없이 주석 이미지 일괄 저장

조작:
    1 / 2 / 3   : 이진화 백엔드 (hsv / adaptive / tophat)
    w           : 검출 백엔드 토글 (centroid / sliding)
    d/Space/->  : 다음      a/<-  : 이전
    s           : 자동 재생 토글
    q / ESC     : 종료
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

import bev as bevlib
import binarize
import config as cfg
import control
import detect

PLAY_DELAY_MS = 200
WINDOW = 'Lane Review  (d/a: next/prev, 1/2/3: binarize, w: detect, s: auto, q: quit)'


def gt_error(res, label):
    """검출 결과와 정답 라벨의 오차. (오차, 설명) 을 돌려준다."""
    if label is None:
        return None, ''
    left, right = label.get('left'), label.get('right')

    if left is not None and right is not None:
        if res.center_x is None:
            return None, 'GT: both lanes, MISSED'
        err = abs(res.center_x - (left + right) / 2)
        return err, f'GT center err={err:.1f}px'

    if left is not None or right is not None:
        truth = left if left is not None else right
        pred = res.left_x if left is not None else res.right_x
        side = 'L' if left is not None else 'R'
        if pred is None:
            return None, f'GT: {side} only, MISSED'
        err = abs(pred - truth)
        return err, f'GT {side}-only err={err:.1f}px'

    # 차선 없음이 정답
    if res.center_x is not None:
        return None, 'GT: no lane, but DETECTED (false positive)'
    return 0.0, 'GT: no lane, correctly empty'


def draw_bev(frame, y_start, res, bin_name, det_name, info, label,
             ctrl=None, metric=None):
    """BEV 위에 ROI, 중심선, 검출 결과, 정답 라벨, 조향 오차를 그린다."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    x_center = w // 2

    cv2.rectangle(vis, (0, y_start), (w - 1, h - 1), (255, 255, 0), 2)
    cv2.line(vis, (x_center, y_start), (x_center, h), (0, 0, 255), 2)

    # 검출기별 부산물
    for cnt in res.viz.get('contours', []):
        shifted = cnt.copy()
        shifted[:, 0, 1] += y_start
        cv2.drawContours(vis, [shifted], -1, (255, 255, 255), -1)

    for (x_low, y_low, x_high, y_high) in res.viz.get('boxes', []):
        cv2.rectangle(vis, (x_low, y_low + y_start), (x_high, y_high + y_start),
                      (0, 200, 0), 1)

    roi_h = h - y_start

    # 좌우 다항식 사이를 반투명 초록으로 채워 주행 가능 영역을 보여준다
    if res.fit_left is not None and res.fit_right is not None:
        left_pts = bevlib.curve_points(res.fit_left, 0, roi_h, y_offset=y_start)
        right_pts = bevlib.curve_points(res.fit_right, 0, roi_h, y_offset=y_start)
        band = np.vstack([left_pts, right_pts[::-1]]).astype(np.int32)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [band], (0, 190, 0))
        vis = cv2.addWeighted(overlay, 0.28, vis, 0.72, 0)

    for fit, color in ((res.fit_left, (255, 128, 0)), (res.fit_right, (0, 128, 255))):
        if fit is None:
            continue
        pts = bevlib.curve_points(fit, 0, roi_h, y_offset=y_start).astype(np.int32)
        cv2.polylines(vis, [pts], False, color, 2)

    # 중심선 — 좌우 다항식 계수의 평균. 이게 실제 주행 목표 경로다.
    if res.fit_center is not None:
        pts = bevlib.curve_points(res.fit_center, 0, roi_h, y_offset=y_start).astype(np.int32)
        cv2.polylines(vis, [pts], False, (0, 0, 0), 6)
        cv2.polylines(vis, [pts], False, (0, 255, 255), 3)

    # 정답 라벨 — 속 빈 원
    if label is not None:
        row = label.get('row', cfg.GT_ROW_BEV)
        cv2.line(vis, (0, row), (w - 1, row), (0, 255, 255), 1)
        if label.get('left') is not None:
            cv2.circle(vis, (int(label['left']), row), 13, (255, 0, 0), 2)
        if label.get('right') is not None:
            cv2.circle(vis, (int(label['right']), row), 13, (0, 255, 0), 2)
        if label.get('left') is not None and label.get('right') is not None:
            gt_c = int((label['left'] + label['right']) / 2)
            cv2.circle(vis, (gt_c, row), 15, (0, 255, 255), 2)

    # 검출 결과 — 채운 원
    if res.center_x is not None:
        ey = int(res.eval_y) + y_start
        cv2.circle(vis, (int(res.left_x), ey), 8, (255, 0, 0), -1)
        cv2.circle(vis, (int(res.right_x), ey), 8, (0, 255, 0), -1)
        cv2.circle(vis, (int(res.center_x), ey), 10, (0, 255, 255), -1)

        error, steering = detect.lane_error(res.center_x)
        width_txt = f' width={res.width:.0f}' if res.width is not None else ''
        cv2.putText(vis, f'error={error:.0f}px steering={steering:+.2f}{width_txt}',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(vis, 'Lane not detected', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(vis, f'[{bin_name} + {det_name}] status={res.status}', (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(vis, info, (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    draw_control(vis, ctrl, metric)
    if ctrl is not None and ctrl.ok:
        txt = (f'servo={ctrl.servo}  delta={ctrl.delta_deg:+.1f}deg  '
               f'Ld={ctrl.lookahead_cm:.0f}cm' + ('  [CLAMPED]' if ctrl.clamped else ''))
        cv2.putText(vis, txt, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(vis, txt, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 255), 1)

    _, gt_txt = gt_error(res, label)
    if gt_txt:
        cv2.putText(vis, gt_txt, (20, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(vis, gt_txt, (20, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 120), 1)
    return vis


def draw_original(original, matrix, y_start, res, ctrl=None, metric=None):
    """원본 이미지에 SRC 사각형과 L/R/C 역투영을 그린다."""
    vis = original.copy()
    cv2.polylines(vis, [cfg.get_src_points().astype(np.int32)], True, (255, 255, 0), 2)

    if matrix is None or res.center_x is None:
        return vis

    matrix_inv = np.linalg.inv(matrix)
    roi_h = original.shape[0] - y_start
    draw_control(vis, ctrl, metric, matrix_inv)

    # 중심선을 원본 카메라 시점으로 역투영 — 검출이 맞는지 가장 확실한 확인법
    if res.fit_center is not None:
        pts = bevlib.project_points(
            bevlib.curve_points(res.fit_center, 0, roi_h, y_offset=y_start), matrix_inv)
        cv2.polylines(vis, [pts], False, (0, 0, 0), 6)
        cv2.polylines(vis, [pts], False, (0, 255, 255), 3)

    for fit, color in ((res.fit_left, (255, 128, 0)), (res.fit_right, (0, 128, 255))):
        if fit is None:
            continue
        pts = bevlib.project_points(
            bevlib.curve_points(fit, 0, roi_h, y_offset=y_start), matrix_inv)
        cv2.polylines(vis, [pts], False, color, 2)

    ey = res.eval_y + y_start
    for x, color, tag in ((res.left_x, (255, 0, 0), 'L'),
                          (res.right_x, (0, 255, 0), 'R'),
                          (res.center_x, (0, 255, 255), 'C')):
        px, py = bevlib.project_point((x, ey), matrix_inv)
        cv2.circle(vis, (px, py), 9, color, -1)
        cv2.putText(vis, tag, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def process(path, already_bev, bin_name, det_name, pp=None):
    frame, matrix, original = bevlib.load_bev(path, already_bev=already_bev)
    if frame is None:
        return None
    roi, y_start = bevlib.roi_of(frame)
    mask = binarize.BACKENDS[bin_name](roi)
    res = detect.DETECTORS[det_name](mask)
    ctrl = pp(res, roi.shape[0], y_start) if pp is not None else None
    return frame, matrix, original, roi, y_start, mask, res, ctrl


def arc_bev_points(ctrl, metric):
    """Pure Pursuit 원호를 BEV 픽셀 점열로. 목표점보다 조금 더 길게 그린다."""
    X, Y = control.arc_points(ctrl.kappa, ctrl.lookahead_cm * 1.25, n=60)
    x_px, y_px = bevlib.vehicle_to_bev(X, Y, metric)
    return np.stack([x_px, y_px], axis=1)


def draw_control(vis, ctrl, metric, matrix_inv=None):
    """목표점과 예상 주행 원호를 그린다. matrix_inv 를 주면 원본 시점으로 역투영."""
    if ctrl is None or not ctrl.ok:
        return

    pts = arc_bev_points(ctrl, metric)
    if matrix_inv is not None:
        pts = bevlib.project_points(pts, matrix_inv)
    pts = pts.astype(np.int32)

    cv2.polylines(vis, [pts], False, (0, 0, 0), 6)
    cv2.polylines(vis, [pts], False, (255, 0, 255), 2)

    gx, gy = ctrl.goal_bev
    if matrix_inv is not None:
        gx, gy = bevlib.project_point((gx, gy), matrix_inv)
    cv2.circle(vis, (int(gx), int(gy)), 11, (0, 0, 0), -1)
    cv2.circle(vis, (int(gx), int(gy)), 9, (255, 0, 255), -1)


def _titled(img, text):
    """패널 위에 제목 띠를 얹는다."""
    bar = np.zeros((26, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.vstack([bar, img])


def build_panel(out, already_bev, bin_name, det_name, info, label, scale=1.0):
    """원본 + BEV + 마스크를 한 장으로 합친다. 창 하나로 순회하기 위한 것."""
    frame, matrix, original, _, y_start, mask, res, ctrl = out
    metric = cfg.get_metric()

    bev_vis = draw_bev(frame, y_start, res, bin_name, det_name, info, label, ctrl, metric)
    panels = []
    if not already_bev:
        panels.append(_titled(draw_original(original, matrix, y_start, res, ctrl, metric),
                              'ORIGINAL + back-projected'))
    panels.append(_titled(bev_vis, "BIRD'S EYE + centerline"))

    top = np.hstack(panels)

    # 마스크는 원래 비율 그대로 두고 좌우를 검게 채운다.
    # 위쪽 폭에 맞춰 늘리면 세로가 2배가 되어 화면을 잡아먹고, 가로세로가
    # 왜곡되면 차선 기울기를 눈으로 판단하기 어려워진다.
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    pad = top.shape[1] - mask_bgr.shape[1]
    if pad > 0:
        left = pad // 2
        mask_bgr = cv2.copyMakeBorder(mask_bgr, 0, 0, left, pad - left,
                                      cv2.BORDER_CONSTANT, value=(40, 40, 40))
    elif pad < 0:
        mask_bgr = cv2.resize(mask_bgr, (top.shape[1],
                                         int(mask_bgr.shape[0] * top.shape[1] / mask.shape[1])))

    panel = np.vstack([top, _titled(mask_bgr, f'MASK (ROI) - {bin_name}')])

    if scale != 1.0:
        panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='captures',
                    help="'captures' | 'captures_bev' | 임의 폴더 경로 "
                         "(drive.py --record 로 저장한 주행 프레임 등)")
    # 기본값을 GT 평가 승자(중심 MAE 1.9px)로 둔다. 플래그 없이 바로 쓸 수 있게.
    ap.add_argument('--binarize', default='adaptive', choices=list(binarize.BACKENDS))
    ap.add_argument('--detect', default='sliding', choices=list(detect.DETECTORS))
    ap.add_argument('--sort', default='name', choices=['name', 'worst'],
                    help='worst=정답 라벨 대비 오차 큰 순 (gt.json 필요)')
    ap.add_argument('--scale', type=float, default=0.75, help='창 크기 배율')
    ap.add_argument('--lookahead', type=float, default=cfg.LOOKAHEAD_CM,
                    help='Pure Pursuit look-ahead 거리 (cm)')
    ap.add_argument('--no-control', action='store_true', help='Pure Pursuit 오버레이 끄기')
    ap.add_argument('--export', metavar='DIR',
                    help='GUI 대신 주석 이미지를 이 폴더에 저장')
    args = ap.parse_args()

    if args.src == 'captures_bev':
        directory, already_bev = cfg.CAPTURES_BEV_DIR, True
    elif args.src == 'captures':
        directory, already_bev = cfg.CAPTURES_DIR, False
    else:
        directory, already_bev = args.src, False      # 임의 폴더는 원본으로 취급
    paths = sorted(
        p for p in glob.glob(os.path.join(directory, '*'))
        if p.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if not paths:
        raise SystemExit(f'이미지가 없습니다: {directory}')

    metric = cfg.get_metric()
    pp = None if args.no_control else control.PurePursuit(
        metric=metric, lookahead_cm=args.lookahead)
    if pp is not None and not metric.measured:
        print('[경고] metric.json 이 없어 종방향 스케일이 추정값입니다. '
              'servo 절대값은 신뢰하지 마세요 (calibrate_metric.py).')

    gt = {}
    if os.path.exists(cfg.GT_PATH):
        with open(cfg.GT_PATH, encoding='utf-8') as f:
            gt = json.load(f)

    bin_names = list(binarize.BACKENDS)
    det_names = list(detect.DETECTORS)
    bin_idx = bin_names.index(args.binarize)
    det_idx = det_names.index(args.detect)

    if args.sort == 'worst':
        if not gt:
            raise SystemExit('--sort worst 는 gt.json 이 필요합니다. label_gt.py 를 먼저 돌리세요.')

        def rank(p):
            out = process(p, already_bev, args.binarize, args.detect, pp)
            if out is None:
                return -1
            err, _ = gt_error(out[6], gt.get(cfg.bev_key(os.path.basename(p), already_bev)))
            return 1e9 if err is None else err   # 미검출/오검출을 맨 앞으로

        paths.sort(key=rank, reverse=True)

    if not args.export:
        print(__doc__)
    print(f'소스: {args.src} ({len(paths)}장)  라벨: {len(gt)}장  정렬: {args.sort}')

    # ---- 일괄 저장 모드 ----
    if args.export:
        os.makedirs(args.export, exist_ok=True)
        for order, path in enumerate(paths):
            out = process(path, already_bev, bin_names[bin_idx], det_names[det_idx], pp)
            if out is None:
                continue
            name = os.path.basename(path)
            info = f'[{order + 1}/{len(paths)}] {name} blobs={detect.count_blobs(out[5])}'
            panel = build_panel(out, already_bev, bin_names[bin_idx], det_names[det_idx],
                                info, gt.get(cfg.bev_key(name, already_bev)), args.scale)
            cv2.imwrite(os.path.join(args.export, f'{order:03d}_{name}'), panel)
        print(f'저장 완료: {args.export}/ ({len(paths)}장, 파일명 앞 번호가 정렬 순서)')
        return

    # ---- 대화형 모드 ----
    idx = 0
    auto_play = False
    while True:
        path = paths[idx]
        name = os.path.basename(path)
        out = process(path, already_bev, bin_names[bin_idx], det_names[det_idx], pp)
        if out is None:
            idx = (idx + 1) % len(paths)
            continue
        info = (f'[{idx + 1}/{len(paths)}] {name} '
                f'blobs={detect.count_blobs(out[5])} '
                f'({"AUTO" if auto_play else "MANUAL"})')

        cv2.imshow(WINDOW,
                   build_panel(out, already_bev, bin_names[bin_idx], det_names[det_idx],
                               info, gt.get(cfg.bev_key(name, already_bev)), args.scale))

        key = cv2.waitKey(PLAY_DELAY_MS if auto_play else 0) & 0xFF

        if key in (27, ord('q')):
            break
        elif key in (ord('1'), ord('2'), ord('3')):
            pick = key - ord('1')
            if pick < len(bin_names):
                bin_idx = pick
                print(f'  이진화 -> {bin_names[bin_idx]}')
        elif key == ord('w'):
            det_idx = (det_idx + 1) % len(det_names)
            print(f'  검출 -> {det_names[det_idx]}')
        elif key == ord('s'):
            auto_play = not auto_play
            print(f'  {"자동 재생" if auto_play else "수동"} 모드')
        elif key in (ord('d'), ord('n'), 32, 13, 83):
            idx = (idx + 1) % len(paths)
        elif key in (ord('a'), ord('p'), 81):
            idx = (idx - 1 + len(paths)) % len(paths)
        elif auto_play:
            idx = (idx + 1) % len(paths)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
