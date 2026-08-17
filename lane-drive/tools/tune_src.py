"""폭-기울기 지표로 SRC_POINTS를 재튜닝한다.

직선 구간에서 BEV 차선 폭은 y에 무관하게 일정해야 한다. 현재 SRC_POINTS는
근거리로 갈수록 폭이 벌어져(기울기 > 0) 원근이 덜 펴진 상태다.

8자유도를 그대로 풀면 프록시 지표에 과적합되므로 좌우 대칭 사다리꼴
5개 파라미터로 제한한다. scipy가 없으므로 numpy 그리드 + 좌표하강.

    python tune_src.py            # 탐색만 하고 결과 출력 (dry-run)
    python tune_src.py --save     # calib.json 에 기록
"""
import argparse
import glob
import os

import cv2
import numpy as np

import _path  # noqa: F401  — 상위 폴더를 import 경로에 추가 (아래 형제 모듈용)
import bev as bevlib
import binarize
import config as cfg

# 직선 구간 판정: 차선 중심의 y방향 기울기가 이보다 완만하면 직선으로 본다
STRAIGHT_CENTER_SLOPE = 0.30
MIN_ROWS = 6                # 프로파일이 이보다 적으면 측정 실패로 본다
FAIL_PENALTY = 3.0          # 측정 실패한 프레임에 물리는 벌점


def measure(img, src_pts, bin_fn):
    """원본 한 장을 주어진 src로 warp해 (width_slope, center_slope)를 잰다."""
    warped, _ = bevlib.warp_image(img, src_pts)
    ys, widths, centers = bevlib.lane_width_profile(warped, bin_fn)
    if len(ys) < MIN_ROWS:
        return None, None
    return bevlib.width_slope(ys, widths), bevlib.center_slope(ys, centers)


def load_images():
    paths = sorted(
        p for p in glob.glob(os.path.join(cfg.CAPTURES_DIR, '*'))
        if p.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            images.append((os.path.basename(p), cv2.resize(img, (cfg.W, cfg.H))))
    return images


def select_straight(images, src_pts, bin_fn):
    """초기 src 기준으로 직선 구간 프레임을 고른다.

    이 집합은 최적화 내내 고정한다. 후보마다 다시 고르면 최적화가
    "측정 안 되는 프레임을 떨어뜨려" 목적함수를 낮추는 편법을 쓸 수 있다.
    """
    chosen = []
    for name, img in images:
        w_slope, c_slope = measure(img, src_pts, bin_fn)
        if w_slope is None:
            continue
        if abs(c_slope) <= STRAIGHT_CENTER_SLOPE:
            chosen.append((name, img, w_slope))
    return chosen


def objective(params, frames, bin_fn):
    """직선 프레임들의 |폭 기울기| 평균. 측정 실패는 벌점."""
    src = bevlib.trapezoid_to_src(*params)

    # 화면 밖으로 나가거나 위아래가 뒤집힌 후보는 배제
    if not (0 <= src[:, 0].min() and src[:, 0].max() < cfg.W):
        return 1e6
    if not (0 <= src[:, 1].min() and src[:, 1].max() < cfg.H):
        return 1e6
    if params[1] >= params[3] - 20:      # top_y < bot_y
        return 1e6

    total = 0.0
    for _, img, _ in frames:
        w_slope, _ = measure(img, src, bin_fn)
        total += FAIL_PENALTY if w_slope is None else abs(w_slope)
    return total / len(frames)


def coordinate_descent(init, frames, bin_fn, passes=3):
    """파라미터를 하나씩 그리드 탐색하며 돌아가는 좌표하강.

    폭 발산을 지배하는 것은 top_hw : bot_hw 비율이므로 그 둘을 먼저 본다.
    """
    names = ['cx', 'top_y', 'top_hw', 'bot_y', 'bot_hw']
    order = [2, 4, 1, 3, 0]              # top_hw, bot_hw, top_y, bot_y, cx
    spans = {0: 20.0, 1: 30.0, 2: 80.0, 3: 30.0, 4: 80.0}

    best = list(init)
    best_score = objective(best, frames, bin_fn)
    print(f'  초기 objective = {best_score:.4f}')

    for p in range(passes):
        shrink = 0.5 ** p
        for i in order:
            span = spans[i] * shrink
            candidates = best[i] + np.linspace(-span, span, 9)
            local_best, local_score = best[i], best_score
            for value in candidates:
                trial = list(best)
                trial[i] = float(value)
                score = objective(trial, frames, bin_fn)
                if score < local_score:
                    local_best, local_score = float(value), score
            if local_score < best_score:
                best[i], best_score = local_best, local_score
        print(f'  pass {p + 1}: objective = {best_score:.4f}  '
              + '  '.join(f'{n}={v:.1f}' for n, v in zip(names, best)))

    return best, best_score


def report_slopes(images, src_pts, bin_fn, label):
    slopes = []
    for _, img in images:
        w_slope, _ = measure(img, src_pts, bin_fn)
        if w_slope is not None:
            slopes.append(w_slope)
    slopes = np.array(slopes)
    print(f'  {label}: n={len(slopes)}  median={np.median(slopes):+.4f}  '
          f'mean|slope|={np.abs(slopes).mean():.4f}  '
          f'양수={int((slopes > 0).sum())}/{len(slopes)}')
    return slopes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binarize', default=binarize.BASELINE, choices=list(binarize.BACKENDS))
    ap.add_argument('--save', action='store_true', help='결과를 calib.json 에 기록')
    args = ap.parse_args()

    bin_fn = binarize.BACKENDS[args.binarize]
    images = load_images()
    print(f'원본 {len(images)}장 로드 (binarize={args.binarize})\n')

    init_src = cfg.SRC_POINTS_DEFAULT
    init_params = bevlib.src_to_trapezoid(init_src)

    print('[1] 직선 구간 프레임 선별')
    frames = select_straight(images, init_src, bin_fn)
    print(f'  |center slope| <= {STRAIGHT_CENTER_SLOPE} 인 프레임 {len(frames)}장 선정')
    if len(frames) < 5:
        raise SystemExit('직선 프레임이 너무 적어 튜닝이 의미 없습니다.')

    print('\n[2] 튜닝 전 지표 (전체 프레임)')
    report_slopes(images, init_src, bin_fn, 'before')

    print('\n[3] 좌표하강 탐색')
    best_params, best_score = coordinate_descent(init_params, frames, bin_fn)
    best_src = bevlib.trapezoid_to_src(*best_params)

    print('\n[4] 튜닝 후 지표 (전체 프레임)')
    report_slopes(images, best_src, bin_fn, 'after ')

    print('\n[5] SRC_POINTS')
    print('  before:', np.round(init_src).astype(int).tolist())
    print('  after :', np.round(best_src).astype(int).tolist())

    if args.save:
        path = cfg.save_calib(
            best_src,
            note=f'tune_src.py binarize={args.binarize} objective={best_score:.4f} '
                 f'straight_frames={len(frames)}',
        )
        print(f'\n저장했습니다: {path}')
    else:
        print('\n(dry-run — 반영하려면 --save)')


if __name__ == '__main__':
    main()
