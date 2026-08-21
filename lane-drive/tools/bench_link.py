"""원격 추론 링크 품질 측정 — 대회 전에 트랙 현장에서 돌린다.

실제 주행에 쓰는 것과 같은 크기의 JPEG 페이로드를 N회 왕복시켜 지연 분포를 낸다.

    python3 tools/bench_link.py --host 192.168.2.1:5010 -n 1000

**ping 으로는 부족하다.** ping 은 64바이트 패킷의 지연이고 우리가 보내는 건
40KB 다. 크기가 다르면 재전송·단편화 거동이 달라 숫자가 안 맞는다.

**보는 것은 평균이 아니라 p99 다.** 대회장 공용 WiFi 는 p50 이 5ms 여도 p99 가
300ms 씩 튀고, 그 한 번이 하필 신호등 앞이면 끝이다. 평균만 보면 이걸 놓친다.

**왕복에서 추론 시간을 빼야 순수 네트워크 지연이 보인다.** 서버가 자기가 쓴
추론 시간(infer_ms)을 함께 돌려주므로 여기서 나눠 찍는다. 느린 원인이
네트워크인지 맥의 추론인지를 구분하지 못하면 엉뚱한 것을 고치게 된다.

판단 기준: **p99 가 200ms 를 넘으면 이 방식은 위험하다.** 전송 방식(생 TCP,
ZeroMQ)을 손대기 전에 네트워크 구성부터 의심할 것 — 공용 WiFi 를 쓰고 있지
않은지, AP 가 Pi 와 맥 사이에 몇 홉인지.
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

import _path  # noqa: F401
import config as cfg


def percentile(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else float('nan')


def main():
    ap = argparse.ArgumentParser(description='원격 추론 링크 왕복 지연 측정')
    ap.add_argument('--host', required=True, metavar='HOST[:PORT]',
                    help='맥의 추론 서버 주소')
    ap.add_argument('-n', '--count', type=int, default=200, help='왕복 횟수')
    ap.add_argument('--quality', type=int, default=cfg.YOLO_JPEG_QUALITY,
                    help=f'JPEG 품질 (기본 {cfg.YOLO_JPEG_QUALITY} — 주행과 같은 값이어야 한다)')
    ap.add_argument('--timeout', type=float, default=cfg.YOLO_TIMEOUT_S,
                    help=f'왕복 타임아웃(초, 기본 {cfg.YOLO_TIMEOUT_S})')
    ap.add_argument('--image', metavar='PATH',
                    help='보낼 이미지. 생략하면 obstacles/ 에서 하나 고른다')
    ap.add_argument('--interval', type=float, default=0.0,
                    help='요청 사이 대기(초). 0 이면 최대 속도로 연달아 보낸다')
    args = ap.parse_args()

    # 실제 카메라 프레임을 보내야 크기가 주행과 같아진다. 합성 노이즈 이미지는
    # JPEG 이 안 눌려 훨씬 커지므로 측정이 비관적으로 나온다.
    path = args.image
    if path is None:
        cand = sorted(glob.glob(os.path.join(cfg.OBSTACLES_DIR, '*.jpg')))
        if not cand:
            raise SystemExit('보낼 이미지를 찾지 못했습니다. --image 로 지정하세요.')
        path = cand[len(cand) // 2]

    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f'이미지를 읽을 수 없습니다: {path}')
    img = cv2.resize(img, (cfg.W, cfg.H))

    import yolo_remote
    det = yolo_remote.RemoteDetector(args.host, args.quality, args.timeout,
                                     watchdog_ms=0)

    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
    print(f'  이미지 {os.path.basename(path)}  품질 {args.quality} '
          f'-> {len(buf) / 1024:.1f} KB   {args.count}회 왕복\n')

    rtts, infers = [], []
    fails = 0
    t0 = time.monotonic()
    for i in range(args.count):
        before = det.runs
        det.infer(img)
        if det.runs == before:
            fails += 1
        else:
            link = det.link
            rtts.append(link['rtt_ms'])
            infers.append(link['infer_ms'])
        if args.interval:
            time.sleep(args.interval)
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{args.count} ...')
    elapsed = time.monotonic() - t0

    if not rtts:
        raise SystemExit(f'\n왕복이 한 번도 성공하지 못했습니다 ({fails}회 모두 실패). '
                         f'\n  마지막 오류: {det.last_error}')

    rtts, infers = np.array(rtts), np.array(infers)
    nets = rtts - infers                # 순수 네트워크 왕복
    print()
    print('=' * 60)
    print(f'  성공 {len(rtts)}회  실패 {fails}회 ({100 * fails / args.count:.1f}%)  '
          f'{len(rtts) / max(elapsed, 1e-6):.1f} 회/초')
    for label, arr in (('왕복 전체', rtts), ('  ├ 맥 추론', infers),
                       ('  └ 네트워크', nets)):
        print(f'  {label:10s} ms  p50={percentile(arr, 50):6.1f}  '
              f'p95={percentile(arr, 95):6.1f}  p99={percentile(arr, 99):6.1f}  '
              f'max={arr.max():6.1f}')
    print('=' * 60)

    p99 = percentile(rtts, 99)
    if fails:
        print(f'\n  실패가 {fails}회 있습니다. --timeout({args.timeout}s)이 짧거나 '
              f'링크가 불안정합니다.')
    # 왕복의 어느 쪽이 지배적인지 알려준다 — 엉뚱한 것을 고치지 않도록.
    if percentile(infers, 50) > percentile(nets, 50):
        print('\n  지연의 대부분은 맥의 추론입니다. 네트워크는 병목이 아닙니다.')
        if 'ncnn' not in det.model_name.lower():
            print('  모델을 NCNN 으로 내보내면 맥 CPU 기준 2.19배 빨라집니다:')
            print('    yolo export model=<모델>.pt format=ncnn imgsz=640')
    else:
        print('\n  지연의 대부분은 네트워크입니다. 모델을 바꿔도 나아지지 않습니다 —')
        print('  전용 AP 로 옮기는 것이 유일한 해법입니다.')
    if p99 > 200:
        print('\n  [경고] p99 가 200ms 를 넘습니다. 이 링크로 주행하는 것은 위험합니다.')
        print('         전송 방식을 손대기 전에 네트워크 구성을 먼저 의심하세요 —')
        print('         공용 WiFi 를 쓰고 있지 않은지, Pi 와 맥이 같은 AP 에 1홉으로')
        print('         붙어 있는지 확인할 것.')
    elif p99 > 3 * percentile(rtts, 50):
        print('\n  p50 대비 p99 가 3배 이상입니다 — 지연이 고르지 않습니다.')
        print('  전용 AP(맥 인터넷 공유 또는 여행용 라우터)로 옮기면 대개 잡힙니다.')
    else:
        print('\n  지연이 고릅니다. 이 링크는 쓸 만합니다.')
    print(f'\n  참고: --yolo-watchdog-ms 는 (yolo_every / fps * 1000 + 왕복 p99) 보다')
    print(f'        충분히 커야 정상 주행 중에 걸리지 않습니다.')


if __name__ == '__main__':
    main()
