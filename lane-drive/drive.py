"""라즈베리파이 실시간 주행 루프 + 웹 디버그 대시보드 — 진입점.

    afb1.camera.get_image()
      -> 채널 스왑 -> BEV warp -> 이진화 -> 슬라이딩 윈도우 -> 중심선
      -> Pure Pursuit -> afb1.gpio.servo() / motor()

이 파일은 인자 파싱과 조립만 한다. 실제 동작은 아래 네 모듈에 있다.

    hardware.py  Afb1Hardware / ReplayHardware — read/servo/motor/shutdown
    driver.py    Driver — 주행 상태 기계 (AUTO / MANUAL / STOP)
    loop.py      Shared / Profiler / run_loop / print_summary
    webui.py     PAGE / make_app — Flask 대시보드 (HUD 그리기는 hud.py)

디버그 화면은 **자체 Flask 서버**로 띄운다 (raspi/L_5_Capture.py 방식).
afb1.flask 는 쓰지 않는다 — 동작을 확인할 수 없어 화면이 안 뜰 위험이 있고,
직접 띄우면 주소와 포트를 우리가 안다.

    http://<Pi주소>:5000

사용:
    python3 drive.py --dry-run                      # 모터 끈 채 확인 (첫 브링업)
    python3 drive.py --speed 40 --record run1       # 주행 + 녹화 (웹 자동 활성)
    python3 drive.py --replay ../project/captures   # 하드웨어 없이 로직 검증

주행 모드 (웹에서 전환):
    AUTO   — Pure Pursuit 자율주행
    MANUAL — 방향키/화면 버튼으로 직접 조종 (인지는 계속 돌아 화면은 살아 있다)
    STOP   — 모터 정지

안전:
    - 웹이 켜져 있으면 STOP 으로 시작한다. 실행하자마자 차가 나가지 않는다
    - AUTO 속도는 --speed 를 명시해야 0 이 아니다
    - 웹에서 STOP / 비상정지 버튼으로 즉시 세울 수 있다
    - 연속 MAX_FAIL_FRAMES 프레임 검출 실패 시 자동 정지
    - Ctrl+C / 예외 / 정상 종료 어느 쪽이든 finally 에서 stop_all()
"""
import argparse
import dataclasses
import os
import threading
import time

import cv2

import binarize
import config as cfg
import control
import detect
import driver as driverlib
import hardware
import loop
import webui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=int, default=0,
                    help=f'모터 속도 (기본 0 = 정지). 주행하려면 명시. 권장 {cfg.DRIVE_SPEED}')
    ap.add_argument('--dry-run', action='store_true',
                    help='서보만 움직이고 모터는 항상 0 — 첫 브링업용')
    ap.add_argument('--binarize', default='adaptive', choices=list(binarize.BACKENDS))
    ap.add_argument('--detect', default='sliding', choices=list(detect.DETECTORS))
    ap.add_argument('--lookahead', type=float, default=cfg.LOOKAHEAD_CM)
    ap.add_argument('--max-fail', type=int, default=cfg.MAX_FAIL_FRAMES)
    ap.add_argument('--ema', type=float, default=cfg.SERVO_EMA_ALPHA,
                    help='서보 평활 계수. 1.0 = 평활 없음')
    ap.add_argument('--manual-speed', type=int, default=cfg.DRIVE_SPEED,
                    help='MANUAL 모드에서 전진/후진 속도')
    # 디버그 화면
    ap.add_argument('--no-web', action='store_true', help='웹 대시보드 끄기')
    ap.add_argument('--port', type=int, default=5000)
    ap.add_argument('--jpeg-quality', type=int, default=50, help='스트리밍 JPEG 품질')
    ap.add_argument('--window', action='store_true',
                    help='cv2.imshow 창도 띄우기 (Pi에 모니터/VNC 가 있을 때)')
    # 튜닝
    ap.add_argument('--invert-servo', action='store_true',
                    help='서보가 반대로 돌 때. --dry-run 으로 확인 후 사용')
    ap.add_argument('--steer-gain', type=float, default=cfg.SERVO_PER_DEG,
                    help='서보단위/바퀴각(deg). 조향이 과하면 줄이고 모자라면 키운다')
    ap.add_argument('--center-offset', type=float, default=0.0,
                    help='BEV상 차량 중심선 보정(px). + 면 좌선회를 더 한다 '
                         '= 차가 오른쪽으로 쏠릴 때 쓴다')
    # 객체 탐지 (기본 꺼짐 — 주지 않으면 ultralytics 를 import 조차 하지 않는다)
    ap.add_argument('--yolo', action='store_true',
                    help='YOLO 객체 탐지를 켠다. 결과는 화면 상태표에만 표시되고 '
                         '주행 제어에는 관여하지 않는다')
    # 모델 관련 이름은 watch.py 와 맞춘다 (--model / --conf / --imgsz).
    # 주기만 --yolo-every 로 남긴다 — 이 파일의 --detect 는 차선 검출기를
    # 고르는 옵션이라 --detect-every 로 쓰면 헷갈린다.
    ap.add_argument('--model', default=cfg.YOLO_MODEL_PATH, metavar='PATH',
                    help='가중치 경로. .pt 파일 또는 NCNN 내보내기 폴더')
    ap.add_argument('--conf', type=float, default=cfg.YOLO_CONF,
                    help='신뢰도 임계값')
    ap.add_argument('--imgsz', type=int, default=cfg.YOLO_IMGSZ,
                    help='추론 입력 크기. NCNN 모델은 내보낼 때 고정되므로 '
                         '여기서 바꿔도 통하지 않는다')
    ap.add_argument('--yolo-every', type=int, default=cfg.YOLO_EVERY,
                    help='N프레임마다 추론. 동기 실행이라 그 프레임에서 루프가 멈춘다')
    # 데이터
    ap.add_argument('--record', metavar='DIR',
                    help='주행 프레임을 저장. 나중에 review.py 로 되돌려 본다')
    ap.add_argument('--record-every', type=int, default=1, help='N프레임마다 저장')
    ap.add_argument('--replay', metavar='DIR', help='하드웨어 대신 저장 이미지 사용')
    ap.add_argument('--replay-loop', action='store_true', help='replay 반복')
    ap.add_argument('--log-every', type=int, default=30, help='N프레임마다 상태 출력')
    ap.add_argument('--profile', action='store_true',
                    help='단계별 소요시간(ms) 출력 — FPS 저하 원인 추적용')
    ap.add_argument('--pace-fps', type=float, default=cfg.CAMERA_FPS,
                    help='목표 fps. 0 이면 sleep 없이 최대 속도')
    args = ap.parse_args()

    # 웹은 replay 여부와 무관하게 기본으로 뜬다 — replay 로도 대시보드를 그대로
    # 확인할 수 있어야 하드웨어 없이 화면까지 검증된다. 끄려면 --no-web.
    args.web = not args.no_web
    speed = 0 if args.dry_run else args.speed

    metric = cfg.get_metric()
    if not metric.measured:
        print('[경고] metric.json 이 없습니다. 종방향 스케일이 추정값이라 조향각이')
        print('       실제와 다를 수 있습니다. calibrate_metric.py 를 먼저 돌리세요.')

    if args.center_offset:
        metric = dataclasses.replace(
            metric, vehicle_center_x_px=metric.vehicle_center_x_px + args.center_offset)
        print(f'[보정] vehicle_center_x_px = {metric.vehicle_center_x_px:.1f}')

    # 하드웨어보다 먼저 만든다. 모델 파일이 없거나 ultralytics 가 없으면
    # gpio/카메라를 건드리기 전에 끝내야 정리 없이 죽는 일이 없다.
    det = None
    if args.yolo:
        # import 도 여기서만 한다 — ultralytics 가 없는 Pi 에서도 --yolo 를
        # 안 주면 주행은 그대로 돌아야 한다
        import yolo
        det = yolo.Detector(args.model, args.conf, args.imgsz)
        print(f'[yolo] {os.path.basename(args.model)} '
              f'imgsz={args.imgsz} conf={args.conf} '
              f'매 {args.yolo_every}프레임  클래스 {len(det.names)}종')

    if args.replay:
        hw = hardware.ReplayHardware(args.replay, loop=args.replay_loop)
        print(f'[replay] {len(hw.paths)}장 — 하드웨어 없이 루프 로직만 검증')
    else:
        hw = hardware.Afb1Hardware()

    pp = control.PurePursuit(metric=metric, lookahead_cm=args.lookahead,
                             servo_per_deg=args.steer_gain)
    # 웹이 있으면 STOP 으로 시작한다 — 실행하자마자 차가 나가지 않게.
    # 웹이 없으면 켤 방법이 없으므로 AUTO 로 시작한다.
    initial_mode = 'STOP' if args.web else 'AUTO'
    driver = driverlib.Driver(hw, pp,
                              binarize.BACKENDS[args.binarize],
                              detect.DETECTORS[args.detect],
                              speed, args.max_fail, args.ema, args.invert_servo,
                              manual_speed=args.manual_speed, mode=initial_mode)
    shared = loop.Shared()
    prof = loop.Profiler(args.profile)

    if args.record:
        os.makedirs(args.record, exist_ok=True)
        print(f'[record] {args.record}/ 에 저장 (매 {args.record_every}프레임)')

    print(f'binarize={args.binarize} detect={args.detect} Ld={args.lookahead}cm '
          f'speed={speed} manual_speed={args.manual_speed} ema={args.ema} '
          f'steer_gain={args.steer_gain:.2f}')
    print(f'시작 모드: {initial_mode}')
    if speed == 0:
        print('AUTO 속도 0 — 조향만 계산합니다. 자율주행하려면 --speed 를 주세요.')

    t0 = time.time()

    if not args.web:
        loop.run_loop(hw, driver, shared, args, prof, det)
    else:
        app = webui.make_app(shared, driver,
                             save_dir=os.path.join(
                                 os.path.dirname(os.path.abspath(__file__)),
                                 'captures'))
        worker = threading.Thread(target=loop.run_loop,
                                  args=(hw, driver, shared, args, prof, det),
                                  daemon=True)
        worker.start()

        print()
        print('=' * 52)
        print(f'  웹 디버그 화면: http://<Pi주소>:{args.port}')
        print('  AUTO / MANUAL / STOP 전환, 비상정지, 캡처 버튼 제공')
        print('  MANUAL 에서는 방향키(또는 화면 버튼)로 직접 조종합니다')
        print('  안전을 위해 STOP 으로 시작합니다 — 웹에서 AUTO 를 누르세요')
        print('=' * 52)
        print('Ctrl+C 로 종료\n')
        try:
            app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
        except KeyboardInterrupt:
            print('\n사용자 종료')
        finally:
            shared.running = False
            worker.join(timeout=2.0)

    if args.window:
        cv2.destroyAllWindows()
    hw.shutdown()
    loop.print_summary(driver, hw, time.time() - t0)


if __name__ == '__main__':
    main()
