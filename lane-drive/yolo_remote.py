"""원격 YOLO 추론 클라이언트 — Pi 에서 돌고, 추론은 맥이 한다.

Pi4 로컬 추론이 너무 느려서, 프레임을 JPEG 로 눌러 맥의 yolo_server.py 로
보내고 박스 목록만 받는다.

    det = RemoteDetector('192.168.2.1:5010')
    det.infer(frame)        # -> 'human 1, red 1'
    det.summary             # 마지막 결과 (왕복이 실패해도 직전 값이 유지된다)
    det.link                # {'ok', 'rtt_ms', 'age_ms', 'misses', 'runs'}

**yolo.Detector 와 같은 표면을 갖는다.** infer / summary / counts / boxes /
total / runs / names 가 같으므로 loop.py 와 watch.py 의 Worker(det.infer, ...)
구성이 그대로 돌아간다. 원격이냐 로컬이냐를 아는 곳은 drive.py 의 조립부뿐이다.

설계 근거 세 가지.

**1) 동기 호출로 충분하다.** loop.Worker 가 한 칸짜리 큐 + 밀리면 버리는 구조라
이 함수는 워커 스레드에서 한 번에 하나씩만 불린다. 여기서 블로킹해도 제어
루프는 다른 스레드에서 계속 돈다.

**2) 대역폭이 아니라 지연이 문제다.** 품질 85 JPEG 이 평균 46KB 라 초당 15회
보내도 5Mbps 수준이다. 최적화 대상은 전송량이 아니라 왕복 지연의 흔들림이고,
그래서 requests.Session 으로 연결을 유지한다 — 세션 없이 매번 POST 하면
요청마다 TCP 핸드셰이크가 붙어 왕복이 하나 더 는다.

**3) 왕복 실패로 죽지 않는다.** 예외를 잡아 misses 만 올리고 직전 결과를
유지한다. 대신 link['age_ms'] 로 결과가 얼마나 묵었는지를 밖에 알려주고,
임계를 넘으면 link['ok'] 가 False 가 된다 (loop.py 가 이걸 보고 모터를 세운다).

**"객체가 없다"와 "결과를 못 받았다"는 다르다.** summary 만 보면 둘이 구분되지
않는다 — 링크가 끊긴 순간을 "빨간불 없음"으로 읽으면 신호를 무시한 채 교차로에
들어간다. 나중에 미션 상태기계를 붙일 때 반드시 link['ok'] 를 함께 봐야 한다.

**프레임의 채널을 건드리지 말 것.** Afb1Hardware.read() 가 이미 COLOR_BGR2RGB
스왑을 한 프레임을 그대로 imencode 하면 맥에서 imdecode 했을 때 채널 순서가
보존된다. 어느 쪽에서든 색변환을 추가하면 yolo.py docstring 의 그 함정
(평균 conf 0.781 -> 0.516)에 빠진다.
"""
import time
from dataclasses import dataclass, field

import cv2

import config as cfg


@dataclass(frozen=True)
class _Snapshot:
    """한 번의 추론 결과. 통째로 원자적으로 교체한다.

    yolo.Detector 는 boxes / counts / summary 를 필드별로 따로 쓰는데, 쓰는 쪽이
    워커 스레드이고 읽는 쪽이 주행 스레드라 지금은 표시용이라 무해해도 제어가
    여기 걸리면 찢어진 상태를 읽을 수 있다. 여기서는 처음부터 하나로 묶는다.
    """
    summary: str = '-'
    counts: dict = field(default_factory=dict)
    boxes: list = field(default_factory=list)
    total: int = 0
    frame_id: int = -1
    sent_mono: float = 0.0      # Pi 시계로 이 프레임을 보낸 시각
    rtt_ms: float = 0.0
    infer_ms: float = 0.0       # 서버가 보고한 추론 시간. rtt 에서 빼면 순수 망 지연


class RemoteDetector:
    """맥의 yolo_server.py 에 추론을 맡기는 Detector."""

    def __init__(self, target, jpeg_quality=None, timeout_s=None, watchdog_ms=None):
        # 'host' 또는 'host:port' 둘 다 받는다
        host, _, port = str(target).partition(':')
        self.base = f'http://{host}:{port or cfg.YOLO_REMOTE_PORT}'
        self.jpeg_quality = cfg.YOLO_JPEG_QUALITY if jpeg_quality is None else jpeg_quality
        self.timeout_s = cfg.YOLO_TIMEOUT_S if timeout_s is None else timeout_s
        self.watchdog_ms = cfg.YOLO_WATCHDOG_MS if watchdog_ms is None else watchdog_ms

        # requests 는 여기서만 import 한다 — 원격을 안 쓰면 없어도 주행이
        # 돌아야 한다 (yolo.py 가 ultralytics 를, webui.py 가 flask 를 지연
        # import 하는 것과 같은 이유).
        try:
            import requests
        except ImportError as exc:
            raise SystemExit(
                'requests 가 없어 원격 탐지를 켤 수 없습니다.\n'
                '  설치: pip install requests   (또는 sudo apt install python3-requests)\n'
                '  로컬 추론으로 돌리려면 --yolo-remote 대신 --yolo 를 쓰세요.'
            ) from exc

        self.sess = requests.Session()
        self.next_id = 0
        self.runs = 0               # 성공한 왕복 수
        self.misses = 0             # 실패한 왕복 수
        self.last_error = ''
        # 시작 시각을 씨앗으로 둔다. 첫 추론이 영영 안 돌아오면 age 가 그대로
        # 자라 워치독에 걸린다 — 그게 맞는 동작이다.
        self._snap = _Snapshot(sent_mono=time.monotonic())

        # /health 로 살아 있는지와 클래스 목록을 확인한다. 여기서 실패하면
        # 하드웨어를 건드리기 전에 끝낸다 (drive.py 가 det 를 hw 보다 먼저
        # 만드는 이유와 같다).
        self.names = self._handshake()

    def _handshake(self):
        try:
            r = self.sess.get(self.base + '/health', timeout=3.0)
            r.raise_for_status()
            info = r.json()
        except Exception as exc:
            raise SystemExit(
                f'추론 서버에 연결할 수 없습니다: {self.base}\n'
                f'  {type(exc).__name__}: {exc}\n'
                '  맥에서 서버가 떠 있는지 확인하세요: python3 yolo_server.py\n'
                '  서버는 --host 0.0.0.0 으로 떠야 하고, 방화벽이 포트를 막지 않아야 합니다.'
            ) from exc

        names = {int(k): v for k, v in info.get('names', {}).items()}
        self.model_name = info.get('model', '')
        print(f'[yolo-remote] {self.base}  모델 {info.get("model")} '
              f'imgsz={info.get("imgsz")} conf={info.get("conf")}  '
              f'클래스 {len(names)}종')
        expected = {'car_red', 'car_white', 'human', 'left', 'red', 'right', 'right_sign'}
        if set(names.values()) != expected:
            print(f'[경고] 서버 클래스가 예상과 다릅니다.')
            print(f'       서버:   {sorted(names.values())}')
            print(f'       예상:   {sorted(expected)}')
        return names

    # -----------------------------
    # Detector 와 같은 표면
    # -----------------------------
    @property
    def summary(self):
        return self._snap.summary

    @property
    def counts(self):
        return self._snap.counts

    @property
    def boxes(self):
        return self._snap.boxes

    @property
    def total(self):
        return self._snap.total

    @property
    def link(self):
        """링크 상태. ok=False 면 결과가 너무 묵었다는 뜻이다."""
        snap = self._snap
        age_ms = (time.monotonic() - snap.sent_mono) * 1000.0
        return {
            'ok': self.watchdog_ms <= 0 or age_ms <= self.watchdog_ms,
            'rtt_ms': snap.rtt_ms,
            'infer_ms': snap.infer_ms,
            'age_ms': age_ms,
            'misses': self.misses,
            'runs': self.runs,
        }

    @property
    def link_str(self):
        """화면 상태표용 한 줄 요약."""
        s = self.link
        head = 'OK' if s['ok'] else 'DOWN'
        return (f"{head} rtt {s['rtt_ms']:.0f}ms  age {s['age_ms']:.0f}ms  "
                f"miss {s['misses']}")

    def infer(self, frame):
        """한 프레임을 서버로 보내고 요약 문자열을 돌려준다.

        frame 은 주행 루프의 것을 그대로 넣는다 (채널 변환 금지 — 모듈 docstring).
        실패해도 예외를 밖으로 내지 않는다 — 직전 요약을 그대로 돌려준다.
        """
        ok, buf = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            self.misses += 1
            self.last_error = 'JPEG 인코딩 실패'
            return self._snap.summary

        frame_id = self.next_id
        self.next_id += 1
        sent = time.monotonic()

        try:
            r = self.sess.post(
                self.base + '/infer', data=buf.tobytes(),
                headers={'Content-Type': 'image/jpeg',
                         'X-Frame-Id': str(frame_id),
                         'X-Sent-Ms': f'{sent * 1000.0:.1f}'},
                timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            # 여기서 죽으면 워커가 죽고 탐지가 영영 멈춘다. 세지기만 한다.
            self.misses += 1
            self.last_error = f'{type(exc).__name__}: {exc}'
            return self._snap.summary

        rtt_ms = (time.monotonic() - sent) * 1000.0
        self._snap = _Snapshot(
            summary=data.get('summary', '-'),
            counts=data.get('counts', {}),
            # JSON 은 리스트로 오지만 draw_boxes 는 언패킹만 하므로 그대로 쓴다
            boxes=[tuple(b) for b in data.get('boxes', [])],
            total=int(data.get('total', 0)),
            frame_id=frame_id,
            sent_mono=sent,
            rtt_ms=rtt_ms,
            infer_ms=float(data.get('infer_ms', 0.0)),
        )
        self.runs += 1
        return self._snap.summary
