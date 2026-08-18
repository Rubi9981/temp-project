#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
[자율주행 경진대회] 교차로(Crossroad) 직진 및 객체 탐지 통합 주행 시스템
==============================================================================
설명:
  1. 검정 도로 위의 흰색 차선 추종 (Pure Pursuit 자율주행).
  2. 교차로 진입 시 차선이 끊어지고 7대 객체가 미탐지되면 직진 모드 (속도 50) 진입.
  3. 위험 객체(red 신호, 보행자, 차량 등) 감지 시 자동 안전 정지.
  4. 웹 대시보드(http://<IP>:5000)를 통한 실시간 모니터링 및 제어.
=============================================================================="""

# ==============================================================================
# ★★★ [사용자 튜닝 파라미터 / 전체 설정 변수] ★★★
# 사용자는 아래의 변수들을 직접 확인하고 수정할 수 있습니다.
# ==============================================================================

# 1. 주행 속도 설정 (0 ~ 100)
SPEED_CROSSROAD        = 50          # [핵심] 교차로 직진 통과 속도
SPEED_NORMAL           = 100         # 일반 차선 추종 주행 속도
SPEED_MANUAL           = 100         # 수동 조작 모드 속도
SPEED_STOP             = 0           # 정지 속도

# 2. 서보 모터 및 조향 설정 (각도 단위)
SERVO_CENTER           = 90          # 직진(중립) 서보 각도
SERVO_MIN              = 30          # 좌회전 최대 각도
SERVO_MAX              = 150         # 우회전 최대 각도
SERVO_EMA_ALPHA        = 0.5         # 서보 평활 계수 (0.1 ~ 1.0, 1.0=평활 없음)
INVERT_SERVO           = False       # 서보 방향 반전 여부 (좌우가 반대일 경우 True)

# 3. Pure Pursuit 자율주행 제어기 파라미터
LOOKAHEAD_CM           = 20.0        # 전방 주시 거리 (Lookahead distance, cm)
MAX_STEER_DEG          = 28.0        # 최대 앞바퀴 조향 각도 (deg)
SERVO_PER_DEG          = 60.0 / MAX_STEER_DEG  # 각도당 서보 변위

# 4. 객체 탐지 (YOLO) 설정
# 대상 7개 클래스 목록
TARGET_CLASSES         = ['red', 'left', 'right', 'car_red', 'car_white', 'human', 'right_sign']
# 탐지 시 즉시 정지해야 하는 안전 클래스
SAFETY_STOP_CLASSES    = ['red', 'human', 'car_red', 'car_white']
# 모델 경로 (NCNN 모델 폴더 또는 .pt 파일 경로)
YOLO_MODEL_PATH        = '../object_detection/best_v3_ncnn_model'
YOLO_CONF              = 0.25        # 객체 탐지 신뢰도(Confidence) 임계값
YOLO_IMGSZ             = 640         # 추론 해상도 (NCNN 모델은 640 고정)
YOLO_EVERY             = 3           # N프레임마다 YOLO 추론 (주행 루프 반응성 최적화)

# 5. 교차로 판단 파라미터
CROSSROAD_MAX_FRAMES   = 120         # 교차로 직진 모드 최대 지속 프레임수 (약 4초 초과 시 안전 정지)
MAX_FAIL_FRAMES        = 10          # 일반 주행 중 차선 미검출 시 안전 정지 한계 프레임

# 6. 카메라 및 영상 처리 파라미터
CAMERA_WIDTH           = 640         # 카메라 가로 해상도
CAMERA_HEIGHT          = 480         # 카메라 세로 해상도
CAMERA_FPS             = 30          # 카메라 목표 FPS
ROI_Y_RATIO            = 0.55        # 하단 관심영역(ROI) 시작 비율 (0.55 = 아래쪽 45% 영역)
BINARIZE_METHOD        = 'adaptive'  # 이진화 알고리즘 ('adaptive', 'tophat', 'hsv')
DETECT_METHOD          = 'sliding'   # 차선 검출 알고리즘 ('sliding', 'centroid')

# 7. 웹 대시보드 및 디버그 설정
WEB_DASHBOARD_PORT     = 5000        # 웹 디버그 서버 포트
JPEG_QUALITY           = 50          # 웹 스트리밍 JPEG 화질 (1~100)
LOG_EVERY_FRAMES       = 30          # N프레임마다 터미널 로그 출력

# ==============================================================================
# 시스템 모듈 임포트 및 경로 설정
# ==============================================================================
import argparse
import collections
import dataclasses
import os
import sys
import threading
import time

import cv2
import numpy as np

# 상위 폴더 및 lane-drive 모듈 경로 등록
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)
LANE_DRIVE_DIR = os.path.join(WORKSPACE_DIR, 'lane-drive')
OBJECT_DETECTION_DIR = os.path.join(WORKSPACE_DIR, 'object_detection')

for p in (BASE_DIR, LANE_DRIVE_DIR, WORKSPACE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import bev as bevlib
import binarize
import config as lane_cfg
import control
import detect
import hardware
import hud
import loop
import webui
import yolo

try:
    from crossroad_driver import CrossroadDriver
    import config as cross_cfg
except ImportError:
    from cross_road.crossroad_driver import CrossroadDriver
    import cross_road.config as cross_cfg


# ==============================================================================
# 메인 실행 루프
# ==============================================================================
def run_crossroad_loop(hw, driver, shared, args, prof=None, det=None):
    """교차로 주행 통합 메인 루프."""
    t0 = time.time()
    last_log = 0
    prof = prof or loop.Profiler(False)
    prof_bg = loop.Profiler(prof.enabled)

    stamps = collections.deque(maxlen=31)
    pace = (1.0 / CAMERA_FPS) if CAMERA_FPS > 0 else 0.0
    inline_display = args.window
    workers = []

    # HUD 시각화 함수 정의 (교차로 상태 표시 추가)
    def _display(item):
        roi, y_start, res, ctrl, tel, sub_state, det_boxes = item
        vis = hud.overlay(roi, y_start, res, ctrl, tel)
        
        # 교차로 상태 및 객체 탐지 상태를 상단에 추가 오버레이
        state_color = (0, 255, 0) if sub_state == 'LANE_FOLLOW' else (0, 165, 255)
        if 'STOP' in sub_state:
            state_color = (0, 0, 255)
        cv2.putText(vis, f"STATE: {sub_state}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2, cv2.LINE_AA)

        ok, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with shared.lock:
                shared.jpeg = buf.tobytes()

    disp_w = None
    if args.web and not inline_display:
        disp_w = loop.Worker(_display, 'display', prof_bg)
        workers.append(disp_w)

    yolo_w = None
    if det is not None:
        yolo_w = loop.Worker(det.infer, 'yolo', prof_bg)
        workers.append(yolo_w)

    rec_w = None
    if args.record:
        rec_w = loop.Worker(lambda it: cv2.imwrite(it[0], it[1]), 'record', prof_bg)
        workers.append(rec_w)

    print("\n[주행 시작] 교차로 자율주행 루프가 가동되었습니다.")

    try:
        while shared.running:
            loop_start = time.time()

            # 1. 카메라 프레임 수신
            t = time.perf_counter()
            frame = hw.read()
            prof.add('read', time.perf_counter() - t)
            if frame is None:
                print("프레임 수신 실패 또는 영상 종료.")
                break
            frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

            n = driver.stats['frames'] + 1

            # 2. 비동기 YOLO 객체 탐지 의뢰
            if yolo_w is not None and n % YOLO_EVERY == 0:
                yolo_w.offer(frame)

            # 3. 교차로 및 차선 판단 스텝 수행
            t = time.perf_counter()
            ctrl, roi, res, y_start, sub_state = driver.step(frame, det=det)
            prof.add('step', time.perf_counter() - t)

            # 4. 텔레메트리 업데이트
            stamps.append(time.time())
            fps_inst = ((len(stamps) - 1) / (stamps[-1] - stamps[0])
                        if len(stamps) > 1 and stamps[-1] > stamps[0] else 0.0)
            fps_avg = n / max(time.time() - t0, 1e-6)

            det_summary = det.summary if det is not None else '-'
            det_boxes = det.boxes if det is not None else []

            tel = {
                'mode': driver.mode,
                'sub_state': sub_state,
                'fps': fps_inst,
                'fps_avg': fps_avg,
                'servo': int(round(driver.servo_cmd)),
                'motor': driver.motor_cmd,
                'status': sub_state,
                'halted': driver.stopped,
                'frames': n,
                'objects': det_summary,
            }

            with shared.lock:
                shared.tel = tel
                shared.raw = frame

            # 5. 화면 표시 / 스트리밍
            if inline_display:
                vis = hud.overlay(roi, y_start, res, ctrl, tel)
                cv2.putText(vis, f"STATE: {sub_state}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                if args.web:
                    ok, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if ok:
                        with shared.lock:
                            shared.jpeg = buf.tobytes()
                cv2.imshow('Crossroad Drive', vis)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    print('\n[종료] 사용자가 창을 닫았습니다.')
                    break
            elif disp_w is not None:
                disp_w.offer((roi, y_start, res, ctrl, tel, sub_state, det_boxes))

            # 6. 주행 프레임 녹화
            if rec_w is not None and n % args.record_every == 0:
                rec_w.offer((os.path.join(args.record, f'frame_{n:06d}.jpg'), frame))

            # 7. 주기적 콘솔 로깅
            if LOG_EVERY_FRAMES and n - last_log >= LOG_EVERY_FRAMES:
                last_log = n
                head = f'  [{n:5d}] {fps_inst:4.1f}fps | {driver.mode:6s} | {sub_state:20s} '
                info = f'servo={tel["servo"]:3d} motor={tel["motor"]:3d} obj=[{det_summary}]'
                print(head + info)
                if prof.enabled:
                    print(prof.report())

            # 8. 목표 FPS 페이싱
            if pace and not args.replay:
                t = time.perf_counter()
                time.sleep(max(0.0, pace - (time.time() - loop_start)))
                prof.add('sleep', time.perf_counter() - t)

    except KeyboardInterrupt:
        print('\n[종료] 사용자가 Ctrl+C를 입력하였습니다.')
    except Exception as exc:
        print(f'\n[에러] 주행 루프 예외 발생: {exc}')
        import traceback
        traceback.print_exc()
    finally:
        shared.running = False
        try:
            driver.apply_motor(0)
            driver.apply_servo(SERVO_CENTER)
        except Exception:
            pass
        for w in workers:
            w.stop()


# ==============================================================================
# 메인 함수 (진입점)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="교차로 직진 및 객체 탐지 자율주행 프로그램")
    parser.add_argument('--speed', type=int, default=SPEED_NORMAL,
                        help=f'일반 차선 주행 속도 (기본: {SPEED_NORMAL})')
    parser.add_argument('--cross-speed', type=int, default=SPEED_CROSSROAD,
                        help=f'교차로 직진 주행 속도 (기본: {SPEED_CROSSROAD})')
    parser.add_argument('--dry-run', action='store_true',
                        help='모터를 0으로 고정하고 조향 및 판단만 테스트')
    parser.add_argument('--no-web', action='store_true', help='웹 대시보드 비활성화')
    parser.add_argument('--port', type=int, default=WEB_DASHBOARD_PORT, help='웹 포트')
    parser.add_argument('--window', action='store_true', help='cv2.imshow 창 띄우기')
    parser.add_argument('--replay', metavar='DIR', help='실제 카메라 대신 이미지 폴더 재생')
    parser.add_argument('--replay-loop', action='store_true', help='replay 무한 반복')
    parser.add_argument('--record', metavar='DIR', help='주행 영상 저장 폴더')
    parser.add_argument('--record-every', type=int, default=1, help='N프레임마다 저장')
    parser.add_argument('--profile', action='store_true', help='루프 소요시간 프로파일링')
    parser.add_argument('--model', default=YOLO_MODEL_PATH, help='YOLO 모델 가중치 경로')
    parser.add_argument('--conf', type=float, default=YOLO_CONF, help='YOLO 신뢰도 임계값')
    parser.add_argument('--no-yolo', action='store_true', help='YOLO 객체 탐지 끄기')
    args = parser.parse_args()

    args.web = not args.no_web
    normal_speed = 0 if args.dry_run else args.speed
    cross_speed = 0 if args.dry_run else args.cross_speed

    print("=" * 60)
    print(" [자율주행 경진대회] 교차로 직진 & 객체 탐지 통합 주행 시스템")
    print("=" * 60)
    print(f"  일반 주행 속도       : {normal_speed}")
    print(f"  교차로 직진 속도     : {cross_speed} (speed={cross_speed})")
    print(f"  서보 중립 (직진)     : {SERVO_CENTER}")
    print(f"  7대 탐지 객체        : {TARGET_CLASSES}")
    print(f"  차선 이진화 / 검출   : {BINARIZE_METHOD} / {DETECT_METHOD}")
    print("=" * 60)

    # 1. YOLO 객체 탐지기 초기화
    det = None
    if not args.no_yolo:
        # 모델 경로 확인 (NCNN 폴더 우선, 없으면 .pt 파일 탐색)
        model_path = args.model
        if not os.path.exists(model_path):
            alt_pt = os.path.join(OBJECT_DETECTION_DIR, 'best_v3.pt')
            if os.path.exists(alt_pt):
                model_path = alt_pt
        print(f"[YOLO] 모델 로딩: {model_path}")
        det = yolo.Detector(model_path, conf=args.conf, imgsz=YOLO_IMGSZ)
        print(f"[YOLO] 로드 완료 (클래스: {list(det.names.values())})")

    # 2. 하드웨어 또는 Replay 초기화
    if args.replay:
        hw = hardware.ReplayHardware(args.replay, loop=args.replay_loop)
        print(f"[Replay] 모드 가동: {len(hw.paths)}장 이미지 재생")
    else:
        hw = hardware.Afb1Hardware()
        print("[하드웨어] 라즈베리파이 afb1 하드웨어 연결 완료")

    # 3. Pure Pursuit 제어기 및 주행 드라이버 생성
    metric = lane_cfg.get_metric()
    pp = control.PurePursuit(
        metric=metric,
        lookahead_cm=LOOKAHEAD_CM,
        wheelbase_cm=cross_cfg.WHEELBASE_CM_DEFAULT,
        max_steer_deg=MAX_STEER_DEG,
        servo_per_deg=SERVO_PER_DEG
    )

    initial_mode = 'STOP' if args.web else 'AUTO'
    driver = CrossroadDriver(
        hw=hw,
        pp=pp,
        bin_fn=binarize.BACKENDS[BINARIZE_METHOD],
        det_fn=detect.DETECTORS[DETECT_METHOD],
        speed_normal=normal_speed,
        speed_crossroad=cross_speed,
        target_classes=TARGET_CLASSES,
        safety_stop_classes=SAFETY_STOP_CLASSES,
        servo_center=SERVO_CENTER,
        max_fail=MAX_FAIL_FRAMES,
        ema_alpha=SERVO_EMA_ALPHA,
        invert_servo=INVERT_SERVO,
        manual_speed=SPEED_MANUAL,
        mode=initial_mode
    )

    shared = loop.Shared()
    prof = loop.Profiler(args.profile)

    if args.record:
        os.makedirs(args.record, exist_ok=True)

    t0 = time.time()

    # 4. 웹 서버 구동 또는 단독 루프 실행
    if not args.web:
        run_crossroad_loop(hw, driver, shared, args, prof, det)
    else:
        app = webui.make_app(
            shared, driver,
            save_dir=os.path.join(BASE_DIR, 'captures')
        )
        worker = threading.Thread(
            target=run_crossroad_loop,
            args=(hw, driver, shared, args, prof, det),
            daemon=True
        )
        worker.start()

        print(f"\n★ 웹 대시보드 주소 : http://<라즈베리파이 IP>:{args.port}")
        print("  - 안전을 위해 최초 상태는 'STOP' 모드입니다.")
        print("  - 웹 화면에서 [AUTO] 버튼을 누르면 자율주행이 시작됩니다.")
        print("  - [비상 정지] 또는 Ctrl+C로 언제든 즉시 세울 수 있습니다.\n")

        try:
            app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
        except KeyboardInterrupt:
            print('\n[종료] 웹 서버를 종료합니다.')
        finally:
            shared.running = False
            worker.join(timeout=2.0)

    if args.window:
        cv2.destroyAllWindows()
    hw.shutdown()

    # 주행 결과 요약 출력
    s = driver.stats
    print("\n" + "=" * 60)
    print(" [주행 통계 요약]")
    print(f"  총 처리 프레임     : {s['frames']}")
    print(f"  차선 추종 (정상)   : {s['lane_follow']} 프레임")
    print(f"  교차로 직진 주행   : {s['crossroad']} 프레임")
    print(f"  객체 탐지 정지     : {s['object_stop']} 프레임")
    print(f"  검출 실패/이상     : {s['fail']} 프레임")
    print("=" * 60)


if __name__ == '__main__':
    main()
