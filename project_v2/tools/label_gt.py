"""정답 라벨 클릭 도구 (선택 사항이지만 강력히 권함).

프록시 지표만으로는 백엔드 순위를 확정할 수 없다. 반사광 두 개를 차선으로
오인해도 "정상 검출"로 집계되고, 차선 폭 표준편차도 완전한 방어막이 아니다.
고정된 BEV 한 행에서 좌/우 차선 x를 직접 찍어두면 evaluate.py 가 프록시가
아닌 실제 px 오차를 낸다.

**차선 띠의 가운데를 클릭한다.** 검출기들이 추정하는 값이 띠의 중심이기
때문이다 (centroid는 컨투어 무게중심, sliding은 차선 픽셀 평균 x).
BEV의 라벨 행에서 띠 폭은 중앙값 37px 이므로 경계선을 찍으면 18px쯤 치우친다.

    python label_gt.py
    python label_gt.py --src captures        # 원본에서 warp해 라벨링

조작:
    좌클릭 2번   : 좌 차선 -> 우 차선 (둘 다 보일 때)
    좌클릭 1번 + l : 그 점이 "좌 차선", 우 차선은 안 보임
    좌클릭 1번 + r : 그 점이 "우 차선", 좌 차선은 안 보임
    u            : 현재 프레임 클릭 취소
    s            : 이 행에 차선이 아예 없음
    d / Space    : 다음 (라벨 없이 건너뜀)      a : 이전
    q / ESC      : 저장 후 종료

기록 형식 (gt.json):
    {"row": 372, "left": 120, "right": 550}    둘 다 보임
    {"row": 372, "left": 120, "right": null}   좌측만 보임
    {"row": 372, "left": null, "right": 550}   우측만 보임
    {"row": 372, "left": null, "right": null}  아무것도 없음
"""
import argparse
import glob
import json
import os

import cv2

import _path  # noqa: F401  — 상위 폴더를 import 경로에 추가 (아래 형제 모듈용)
import bev as bevlib
import config as cfg

clicks = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
        clicks.append(x)
        print(f'  클릭 {len(clicks)}: x={x}')


def load_existing():
    if os.path.exists(cfg.GT_PATH):
        with open(cfg.GT_PATH, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save(gt):
    with open(cfg.GT_PATH, 'w', encoding='utf-8') as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)

    both = sum(1 for v in gt.values()
               if v.get('left') is not None and v.get('right') is not None)
    side = sum(1 for v in gt.values()
               if (v.get('left') is None) != (v.get('right') is None))
    none = sum(1 for v in gt.values()
               if v.get('left') is None and v.get('right') is None)
    print(f'\n저장: {cfg.GT_PATH}')
    print(f'  좌우 둘 다 {both}장 / 한쪽만 {side}장 / 차선 없음 {none}장  (총 {len(gt)}장)')


def describe(saved):
    """화면 표시용 문구. cv2.putText 는 Hershey 폰트라 ASCII 만 그릴 수 있다."""
    if saved is None:
        return 'unlabeled'
    left, right = saved.get('left'), saved.get('right')
    if left is not None and right is not None:
        return f'saved  L={left}  R={right}'
    if left is not None:
        return f'saved  L={left}  (LEFT only)'
    if right is not None:
        return f'saved  R={right}  (RIGHT only)'
    return 'saved  NO LANE'


def draw_saved(view, saved, row):
    if saved is None:
        return
    if saved.get('left') is not None:
        cv2.circle(view, (int(saved['left']), row), 7, (255, 0, 0), 2)
    if saved.get('right') is not None:
        cv2.circle(view, (int(saved['right']), row), 7, (0, 255, 0), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='captures_bev', choices=['captures', 'captures_bev'])
    args = ap.parse_args()

    directory = cfg.CAPTURES_BEV_DIR if args.src == 'captures_bev' else cfg.CAPTURES_DIR
    already_bev = args.src == 'captures_bev'
    paths = sorted(
        p for p in glob.glob(os.path.join(directory, '*'))
        if p.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if not paths:
        raise SystemExit(f'이미지가 없습니다: {directory}')

    gt = load_existing()
    row = cfg.GT_ROW_BEV

    cv2.namedWindow('label')
    cv2.setMouseCallback('label', on_mouse)
    print(__doc__)
    if gt:
        print(f'기존 라벨 {len(gt)}장을 이어서 편집합니다.\n')

    idx = 0
    while True:
        path = paths[idx]
        name = os.path.basename(path)
        frame, _, _ = bevlib.load_bev(path, already_bev=already_bev)
        if frame is None:
            idx = (idx + 1) % len(paths)
            continue

        saved = gt.get(name)
        advance = None

        while advance is None:
            view = frame.copy()
            cv2.line(view, (0, row), (cfg.W - 1, row), (0, 255, 255), 1)

            for i, x in enumerate(clicks):
                color = (255, 0, 0) if i == 0 else (0, 255, 0)
                cv2.circle(view, (x, row), 7, color, -1)

            if not clicks:
                draw_saved(view, saved, row)
                state = describe(saved)
                hint = 'click LEFT then RIGHT  |  s:no lane   d/a:next/prev   q:save&quit'
            elif len(clicks) == 1:
                state = f'1 click (x={clicks[0]})'
                hint = 'click again for BOTH  |  l:LEFT only   r:RIGHT only   u:undo'
            else:
                state = '2 clicks'
                hint = ''

            cv2.putText(view, f'[{idx + 1}/{len(paths)}] {name}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(view, state, (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(view, hint, (12, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(view, hint, (12, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
            cv2.imshow('label', view)

            key = cv2.waitKey(20) & 0xFF

            # 두 번 클릭되면 좌우 확정 후 자동으로 다음 장
            if len(clicks) == 2:
                left, right = sorted(clicks)
                gt[name] = {'row': row, 'left': left, 'right': right}
                print(f'  기록: {name} L={left} R={right}')
                advance = +1

            elif key == ord('l') and len(clicks) == 1:
                gt[name] = {'row': row, 'left': clicks[0], 'right': None}
                print(f'  기록: {name} L={clicks[0]} (좌측만)')
                advance = +1

            elif key == ord('r') and len(clicks) == 1:
                gt[name] = {'row': row, 'left': None, 'right': clicks[0]}
                print(f'  기록: {name} R={clicks[0]} (우측만)')
                advance = +1

            elif key == ord('u'):
                clicks.clear()

            elif key == ord('s'):
                gt[name] = {'row': row, 'left': None, 'right': None}
                print(f'  기록: {name} 차선 없음')
                advance = +1

            elif key in (ord('d'), ord('n'), 32, 13, 83):
                advance = +1
            elif key in (ord('a'), ord('p'), 81):
                advance = -1
            elif key in (27, ord('q')):
                cv2.destroyAllWindows()
                save(gt)
                return

        clicks.clear()
        idx = (idx + advance) % len(paths)


if __name__ == '__main__':
    main()
