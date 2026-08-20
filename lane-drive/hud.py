"""주행 상태 HUD — BEV 위에 검출/제어 결과를 그린다.

drive.py 의 웹 스트리밍과 --window 창이 같은 그림을 쓴다.
"""
import cv2
import numpy as np

import bev as bevlib
import config as cfg


def overlay(roi, y_start, res, ctrl, tel):
    """ROI 밴드 위에 주행 상태를 그린다.

    첫 인자는 **분석에 쓴 ROI 밴드**다 (전체 BEV 가 아니다). 위쪽 55%는
    검출에 쓰지 않으므로 애초에 만들지 않는다 — driver.Driver.perceive 참조.
    다항식은 ROI 좌표계라 그대로 그리면 되고, ctrl.goal_bev 만 전체 BEV
    좌표라 y_start 만큼 올려서 찍는다.

    cv2.putText 는 Hershey 폰트라 ASCII 만 그릴 수 있다.
    """
    vis = roi.copy()
    h, w = vis.shape[:2]

    # ROI 테두리 + 화면 중앙 기준선
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (255, 120, 0), 2)
    cv2.line(vis, (w // 2, 0), (w // 2, h), (0, 0, 255), 1)

    # 좌우 차선 다항식
    for fit, color in ((res.fit_left, (255, 128, 0)), (res.fit_right, (0, 128, 255))):
        if fit is not None:
            pts = bevlib.curve_points(fit, 0, h).astype(np.int32)
            cv2.polylines(vis, [pts], False, color, 2)

    # 주행 목표 경로
    if res.fit_center is not None:
        pts = bevlib.curve_points(res.fit_center, 0, h).astype(np.int32)
        cv2.polylines(vis, [pts], False, (0, 0, 0), 6)
        cv2.polylines(vis, [pts], False, (0, 255, 255), 3)

    # Pure Pursuit 목표점 — goal_bev 는 전체 BEV 좌표라 ROI 기준으로 내린다
    if ctrl.ok:
        gx, gy = ctrl.goal_bev
        gy = int(gy) - y_start
        cv2.circle(vis, (int(gx), gy), 10, (0, 0, 0), -1)
        cv2.circle(vis, (int(gx), gy), 8, (255, 0, 255), -1)

    # 좌상단 상태
    mode_color = {'AUTO': (0, 255, 0),
                  'MANUAL': (0, 200, 255),
                  'STOP': (0, 0, 255)}.get(tel['mode'], (200, 200, 200))
    rows = [
        (f"MODE: {tel['mode']}", mode_color),
        (f"FPS: {tel['fps']:.1f}", (200, 200, 200)),
        (f"SERVO: {tel['servo']}", (200, 200, 200)),
        (f"MOTOR: {tel['motor']}", (200, 200, 200)),
    ]
    # 교차로 주행일 때만 붙는다 (CrossroadDriver 의 하위 상태)
    if tel.get('sub_state'):
        sub = tel['sub_state']
        color = ((0, 0, 255) if 'STOP' in sub or 'LOST' in sub or 'FAIL' in sub
                 else (0, 255, 0) if sub == 'LANE_FOLLOW' else (0, 165, 255))
        rows.insert(1, (f'STATE: {sub}', color))
    for i, (text, color) in enumerate(rows):
        y = 28 + i * 26
        cv2.putText(vis, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1,
                    cv2.LINE_AA)

    # 우상단 검출/제어 상태
    if ctrl.ok:
        status = f"{res.status.upper()}  d={ctrl.delta_deg:+.1f}deg  Ld={ctrl.lookahead_cm:.0f}cm"
        status_color = (0, 255, 255)
    else:
        status = 'NO LANE'
        status_color = (0, 0, 255)
    if tel['halted']:
        status = 'HALTED - ' + status
        status_color = (0, 0, 255)
    (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    sx = max(w - tw - 14, 200)      # 좌상단 상태 열과 겹치지 않게
    cv2.putText(vis, status, (sx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(vis, status, (sx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1,
                cv2.LINE_AA)

    # 하단 조향 게이지 — 서보 30~150 을 -100~+100 px 로 매핑
    cx, gy = w // 2, h - 22
    cv2.line(vis, (cx - 100, gy), (cx + 100, gy), (90, 90, 90), 3)
    cv2.line(vis, (cx, gy - 7), (cx, gy + 7), (150, 150, 150), 2)
    # 서보 가동폭이 좌우 비대칭이라(좌 80 / 우 60단위) 한쪽 폭으로 나누면
    # 게이지가 한쪽으로 치우친다. 각 방향의 실제 폭으로 정규화한다.
    delta_servo = tel['servo'] - cfg.SERVO_CENTER
    span = ((cfg.SERVO_CENTER - cfg.SERVO_MIN) if delta_servo < 0
            else (cfg.SERVO_MAX - cfg.SERVO_CENTER))
    offset = int(delta_servo * (100.0 / max(span, 1e-6)))
    offset = int(np.clip(offset, -100, 100))
    cv2.circle(vis, (cx + offset, gy), 9, (0, 0, 0), -1)
    cv2.circle(vis, (cx + offset, gy), 7, (0, 255, 0), -1)
    return vis
