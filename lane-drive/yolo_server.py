"""맥에서 도는 YOLO 추론 서버 — Pi 대신 추론만 대신한다.

Pi4 로컬 추론이 너무 느려서, 카메라 프레임을 맥으로 보내 여기서 추론하고
결과(박스 목록)만 돌려준다. Pi 쪽 클라이언트는 yolo_remote.py 다.

    맥:  python3 yolo_server.py --model ../object_detection/best_v6_ncnn_model
    Pi:  python3 drive.py --yolo-remote <맥주소>:5010 --speed 40

**추론 자체는 yolo.Detector 를 그대로 쓴다.** 박스/카운트/요약 문자열을 만드는
코드가 한 곳뿐이라 로컬 추론과 원격 추론의 결과가 정의상 같아진다. 워밍업(첫
추론의 초기화 비용 제거)도 Detector.__init__ 안에 이미 있어 따로 할 게 없다.

**표준 라이브러리 http.server 를 쓴다 — Flask 가 아니다.** webui.py 는 Flask 로
대시보드를 그리지만 여기는 엔드포인트가 둘뿐이라 라우팅이 필요 없고, 맥에
설치할 것을 하나라도 줄이는 편이 대회장에서 안전하다. keep-alive 는
protocol_version='HTTP/1.1' + 정확한 Content-Length 로 얻는다 — 연결이 유지되어야
요청마다 TCP 핸드셰이크가 붙지 않는다.

**프레임의 채널을 건드리지 말 것.** Pi 의 Afb1Hardware.read() 가 이미
COLOR_BGR2RGB 스왑을 한 프레임을 JPEG 로 보내고, 여기서 imdecode 하면 채널
순서가 그대로 보존된다. 여기서 습관적으로 색변환을 넣으면 yolo.py docstring 의
그 함정(평균 conf 0.781 -> 0.516)에 그대로 빠진다.

맥에서는 .pt 를 그대로 써도 되고 NCNN 폴더를 써도 된다 — YOLO() 가 둘 다 받는다.
맥 CPU 측정으로는 NCNN 이 2.19배 빨랐다 (15.3ms vs 33.5ms).
"""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import config as cfg
import yolo


def make_handler(det, args, stats, lock):
    """요청 핸들러 클래스를 만든다.

    추론은 lock 으로 직렬화한다. ultralytics 모델을 여러 스레드가 동시에
    predict 하는 것은 안전이 보장되지 않고, 어차피 클라이언트(loop.Worker)가
    한 번에 하나씩만 보낸다.
    """

    class Handler(BaseHTTPRequestHandler):
        # keep-alive 를 쓰려면 HTTP/1.1 이어야 하고, 그러면 모든 응답에
        # Content-Length 가 정확히 붙어야 한다 (_json 이 그렇게 한다).
        protocol_version = 'HTTP/1.1'

        def log_message(self, *a):
            """기본 로깅을 끈다 — 초당 15줄씩 찍히면 아무것도 안 보인다."""

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/health':
                return self._json({'ok': False, 'error': 'not found'}, 404)
            # 클라이언트가 시작할 때 1회 호출한다. 클래스 목록을 여기서 받는다.
            self._json({
                'ok': True,
                'names': {str(k): v for k, v in det.names.items()},
                'model': os.path.basename(args.model.rstrip('/')),
                'imgsz': args.imgsz,
                'conf': args.conf,
                'served': stats['served'],
            })

        def do_POST(self):
            if self.path != '/infer':
                return self._json({'ok': False, 'error': 'not found'}, 404)

            # HTTP/1.1 에서는 본문을 Content-Length 만큼 **정확히** 읽어야
            # 다음 요청과 경계가 어긋나지 않는다.
            n = int(self.headers.get('Content-Length', 0))
            data = self.rfile.read(n) if n else b''

            # X-Sent-Ms 는 Pi 가 자기 시계로 찍은 값이라 여기서 해석하지 않는다.
            # 되돌려 보내기만 하면 Pi 가 자기 시계로 나이를 계산할 수 있어서,
            # 맥과 Pi 의 시계를 맞출 필요가 없다.
            frame_id = self.headers.get('X-Frame-Id', '-1')
            sent_ms = self.headers.get('X-Sent-Ms', '0')

            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                stats['bad'] += 1
                return self._json({'ok': False, 'error': 'JPEG 디코딩 실패',
                                   'frame_id': frame_id, 'sent_ms': sent_ms}, 400)

            t = time.perf_counter()
            with lock:
                det.infer(img)
                # 락 안에서 스냅샷을 뜬다 — 다음 요청이 det 를 덮어쓰기 전에
                boxes, counts = det.boxes, det.counts
                summary, total = det.summary, det.total
            infer_ms = (time.perf_counter() - t) * 1000.0
            stats['served'] += 1
            stats['infer_ms'] += infer_ms

            self._json({
                'ok': True,
                'frame_id': frame_id,
                'sent_ms': sent_ms,
                'boxes': boxes,
                'counts': counts,
                'summary': summary,
                'total': total,
                'infer_ms': round(infer_ms, 2),
            })

    return Handler


def main():
    ap = argparse.ArgumentParser(
        description='맥에서 도는 YOLO 추론 서버 (Pi 의 --yolo-remote 상대편)')
    ap.add_argument('--model', default=cfg.YOLO_MODEL_PATH, metavar='PATH',
                    help='가중치 경로. .pt 파일 또는 NCNN 내보내기 폴더')
    ap.add_argument('--conf', type=float, default=cfg.YOLO_CONF)
    ap.add_argument('--imgsz', type=int, default=cfg.YOLO_IMGSZ,
                    help='추론 입력 크기. NCNN 모델은 내보낼 때 고정된다')
    ap.add_argument('--host', default='0.0.0.0',
                    help='0.0.0.0 이어야 Pi 에서 접속할 수 있다')
    ap.add_argument('--port', type=int, default=cfg.YOLO_REMOTE_PORT)
    ap.add_argument('--log-every', type=int, default=100,
                    help='N회 추론마다 상태 출력. 0 이면 조용히')
    args = ap.parse_args()

    det = yolo.Detector(args.model, args.conf, args.imgsz)
    stats = {'served': 0, 'bad': 0, 'infer_ms': 0.0}
    httpd = ThreadingHTTPServer(
        (args.host, args.port), make_handler(det, args, stats, threading.Lock()))
    httpd.daemon_threads = True

    print()
    print('=' * 60)
    print(f'  모델   {args.model}')
    print(f'  클래스 {len(det.names)}종: {", ".join(det.names.values())}')
    print(f'  주소   http://<이 맥의 주소>:{args.port}')
    print(f'  Pi 에서: python3 drive.py --yolo-remote <이 맥의 주소>:{args.port}')
    print('=' * 60)
    print('Ctrl+C 로 종료\n')

    # 주기적 상태 출력. 서버가 살아 있는지, 요청이 실제로 들어오는지를
    # 눈으로 확인할 수 있어야 한다.
    def report():
        last = 0
        while args.log_every:
            time.sleep(2.0)
            n = stats['served']
            if n - last >= args.log_every:
                print(f'  추론 {n}회  평균 {stats["infer_ms"] / n:.1f}ms'
                      + (f'  디코딩 실패 {stats["bad"]}회' if stats['bad'] else ''))
                last = n
    threading.Thread(target=report, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n사용자 종료')
    finally:
        httpd.shutdown()
        print(f'  총 추론 {stats["served"]}회'
              + (f', 디코딩 실패 {stats["bad"]}회' if stats['bad'] else ''))


if __name__ == '__main__':
    main()
