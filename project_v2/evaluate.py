"""데이터셋 전체 벤치마크.

사용 예:
    python evaluate.py --compare-all
    python evaluate.py --binarize hsv --detect centroid --src captures_bev
    python evaluate.py --compare-all --gt

정답 라벨(gt.json)이 없으면 프록시 지표만 나온다. 프록시 지표는
"차선을 제대로 잡았는가"를 직접 재지 못한다 — 반사광 두 개를 차선으로
오인해도 status='ok'가 된다. 차선 폭 표준편차가 그 경우를 잡아내는
보조 장치이지만 완전하지 않다. 확실히 하려면 label_gt.py 를 먼저 돌려라.
"""
import argparse
import glob
import json
import os

import numpy as np

import bev as bevlib
import binarize
import config as cfg
import detect


def collect_paths(src):
    directory = cfg.CAPTURES_BEV_DIR if src == 'captures_bev' else cfg.CAPTURES_DIR
    paths = sorted(
        p for p in glob.glob(os.path.join(directory, '*'))
        if p.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    return paths, (src == 'captures_bev')


def load_gt():
    if not os.path.exists(cfg.GT_PATH):
        return None
    with open(cfg.GT_PATH, encoding='utf-8') as f:
        return json.load(f)


def run(paths, already_bev, binarize_name, detect_name, gt=None):
    """한 조합을 데이터셋 전체에 돌려 통계를 낸다."""
    bin_fn = binarize.BACKENDS[binarize_name]
    det_fn = detect.DETECTORS[detect_name]

    status_count = {'ok': 0, 'single': 0, 'fail': 0}
    blob_hist = {0: 0, 1: 0, 2: 0, 3: 0}      # 3 = "3개 이상"
    widths, errors = [], []
    saturated = 0
    n = 0

    # 정답 라벨 대비 오차. 좌우가 다 보이는 프레임(중심 오차)과 한쪽만 보이는
    # 프레임(해당 차선 위치 오차)은 재는 대상이 달라 따로 집계한다.
    # 단측 프레임의 "중심"은 정답이 정의되지 않는다 — 차선 폭을 가정해야
    # 하는데 그건 검출기가 외삽에 쓰는 것과 같은 값이라 순환 논증이 된다.
    center_err, side_err = [], []
    center_total = center_missed = 0
    side_total = side_missed = 0

    for path in paths:
        frame, _, _ = bevlib.load_bev(path, already_bev=already_bev)
        if frame is None:
            continue
        n += 1

        roi, y_start = bevlib.roi_of(frame)
        mask = bin_fn(roi)

        nb = detect.count_blobs(mask)
        blob_hist[min(nb, 3)] += 1

        res = det_fn(mask)
        status_count[res.status] += 1

        if res.width is not None:
            widths.append(res.width)

        error, _ = detect.lane_error(res.center_x)
        if error is not None:
            errors.append(error)
            if abs(error) >= cfg.STEER_CLIP:
                saturated += 1

        if gt is not None:
            label = gt.get(os.path.basename(path))
            if label is not None:
                gl, gr = label.get('left'), label.get('right')

                if gl is not None and gr is not None:
                    center_total += 1
                    if res.center_x is not None:
                        center_err.append(abs(res.center_x - (gl + gr) / 2))
                    else:
                        center_missed += 1

                elif gl is not None or gr is not None:
                    side_total += 1
                    # 보이는 쪽 차선의 위치만 비교한다
                    truth = gl if gl is not None else gr
                    pred = res.left_x if gl is not None else res.right_x
                    if pred is not None:
                        side_err.append(abs(pred - truth))
                    else:
                        side_missed += 1

    errors = np.array(errors, float)
    widths = np.array(widths, float)
    center_err = np.array(center_err, float)
    side_err = np.array(side_err, float)

    return {
        'binarize': binarize_name,
        'detect': detect_name,
        'n': n,
        'status': status_count,
        'blobs': blob_hist,
        'width_mean': widths.mean() if widths.size else None,
        'width_std': widths.std() if widths.size else None,
        'err_mean': errors.mean() if errors.size else None,
        'err_std': errors.std() if errors.size else None,
        'err_min': errors.min() if errors.size else None,
        'err_max': errors.max() if errors.size else None,
        'saturated': saturated,
        # 좌우 둘 다 보이는 프레임 — 차선 중심 오차
        'center_total': center_total,
        'center_missed': center_missed,
        'center_mae': center_err.mean() if center_err.size else None,
        'center_p95': np.percentile(center_err, 95) if center_err.size else None,
        # 한쪽만 보이는 프레임 — 그 차선의 위치 오차
        'side_total': side_total,
        'side_missed': side_missed,
        'side_mae': side_err.mean() if side_err.size else None,
        'side_p95': np.percentile(side_err, 95) if side_err.size else None,
    }


def _pct(count, total):
    return f'{count:3d} ({100 * count / total:2.0f}%)' if total else '  0 ( 0%)'


def print_detail(r):
    n = r['n']
    print(f"\n{'=' * 62}")
    print(f"  binarize={r['binarize']}  detect={r['detect']}   총 {n}장")
    print('=' * 62)
    print('  [검출 상태]')
    print(f"    정상   (좌우 모두)     : {_pct(r['status']['ok'], n)}")
    print(f"    단측   (한쪽만)        : {_pct(r['status']['single'], n)}")
    print(f"    실패   (검출 없음)     : {_pct(r['status']['fail'], n)}")
    print('  [마스크 덩어리 수 — 이진화 품질, 검출기 무관]')
    print(f"    0개                    : {_pct(r['blobs'][0], n)}")
    print(f"    1개                    : {_pct(r['blobs'][1], n)}")
    print(f"    2개                    : {_pct(r['blobs'][2], n)}")
    print(f"    3개 이상 (과검출)      : {_pct(r['blobs'][3], n)}")

    if r['width_mean'] is not None:
        print('  [차선 폭 — 물리적으로 일정해야 하므로 std가 작을수록 좋다]')
        print(f"    mean={r['width_mean']:.1f}px  std={r['width_std']:.1f}px")

    if r['err_mean'] is not None:
        print('  [조향 error]')
        print(f"    mean={r['err_mean']:.1f}  std={r['err_std']:.1f}  "
              f"범위 {r['err_min']:.0f}~{r['err_max']:.0f}")
        print(f"    clip(±{cfg.STEER_CLIP}) 포화 : {_pct(r['saturated'], n)}")

    if r['center_total'] or r['side_total']:
        print('  [정답 라벨 대비]')
        if r['center_total']:
            mae = f"{r['center_mae']:.1f}" if r['center_mae'] is not None else '-'
            p95 = f"{r['center_p95']:.1f}" if r['center_p95'] is not None else '-'
            print(f"    좌우 둘 다 보임 {r['center_total']}장 — 차선중심 "
                  f"MAE={mae}px p95={p95}px  (검출 실패 {r['center_missed']}장)")
        if r['side_total']:
            mae = f"{r['side_mae']:.1f}" if r['side_mae'] is not None else '-'
            p95 = f"{r['side_p95']:.1f}" if r['side_p95'] is not None else '-'
            print(f"    한쪽만 보임     {r['side_total']}장 — 해당차선 "
                  f"MAE={mae}px p95={p95}px  (검출 실패 {r['side_missed']}장)")


def print_table(results, has_gt):
    header = f"{'binarize':<10}{'detect':<10}{'정상':>7}{'단측':>7}{'실패':>7}{'과검출':>8}{'폭std':>8}{'err std':>9}"
    if has_gt:
        header += f"{'GT중심':>9}{'GT단측':>9}"
    print('\n' + header)
    print('-' * len(header))
    for r in results:
        n = r['n']
        row = (f"{r['binarize']:<10}{r['detect']:<10}"
               f"{100 * r['status']['ok'] / n:6.0f}%"
               f"{100 * r['status']['single'] / n:6.0f}%"
               f"{100 * r['status']['fail'] / n:6.0f}%"
               f"{100 * r['blobs'][3] / n:7.0f}%")
        row += f"{r['width_std']:8.1f}" if r['width_std'] is not None else f"{'-':>8}"
        row += f"{r['err_std']:9.1f}" if r['err_std'] is not None else f"{'-':>9}"
        if has_gt:
            row += f"{r['center_mae']:9.1f}" if r['center_mae'] is not None else f"{'-':>9}"
            row += f"{r['side_mae']:9.1f}" if r['side_mae'] is not None else f"{'-':>9}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='captures',
                    choices=['captures', 'captures_bev'],
                    help='captures=원본에서 매번 warp(기본), captures_bev=기존 BEV 산출물')
    ap.add_argument('--binarize', default=binarize.BASELINE, choices=list(binarize.BACKENDS))
    ap.add_argument('--detect', default=detect.BASELINE, choices=list(detect.DETECTORS))
    ap.add_argument('--compare-all', action='store_true', help='모든 조합 비교')
    ap.add_argument('--gt', action='store_true', help='gt.json 이 있으면 실제 오차도 낸다')
    args = ap.parse_args()

    paths, already_bev = collect_paths(args.src)
    if not paths:
        raise SystemExit(f'이미지가 없습니다: {args.src}')

    gt = load_gt() if args.gt else None
    if args.gt and gt is None:
        print(f'[경고] {cfg.GT_PATH} 가 없어 프록시 지표만 냅니다. '
              f'label_gt.py 를 먼저 돌리세요.')

    print(f"소스: {args.src}  ({len(paths)}장, "
          f"{'이미 BEV' if already_bev else 'SRC_POINTS로 on-the-fly warp'})")

    if args.compare_all:
        results = []
        for b in binarize.BACKENDS:
            for d in detect.DETECTORS:
                results.append(run(paths, already_bev, b, d, gt))
        baseline = next(r for r in results
                        if r['binarize'] == binarize.BASELINE and r['detect'] == detect.BASELINE)
        print_detail(baseline)
        print_table(results, has_gt=gt is not None)
        print('\n* 정상/단측/실패는 검출기 기준, 과검출은 이진화 기준이다.')
        print('* 폭std는 반사광을 차선으로 오인했는지 잡아내는 프록시다 — 낮을수록 좋다.')
        if gt is not None:
            print('* GT중심 = 좌우 둘 다 보이는 프레임의 차선중심 오차(px).')
            print('* GT단측 = 한쪽만 보이는 프레임의 해당 차선 위치 오차(px).')
    else:
        print_detail(run(paths, already_bev, args.binarize, args.detect, gt))


if __name__ == '__main__':
    main()
