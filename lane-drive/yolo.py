"""YOLOv8 객체 탐지 — 주행 화면에 "무엇이 보이는지"를 띄우기 위한 것.

주행 제어에는 관여하지 않는다. 탐지 결과는 상태표에 문자열 한 줄로만 나간다.

    det = Detector(cfg.YOLO_MODEL_PATH, conf=0.25, imgsz=640)
    det.infer(frame)        # -> 'human 1, red 1'
    det.summary             # 마지막 추론 결과 (추론을 건너뛴 프레임에서도 유지)

**입력 프레임을 채널 변환하지 말 것.** best_v3.pt 는 Afb1Hardware.read() 가
COLOR_BGR2RGB 스왑을 한 뒤 저장된 프레임(collect.py 산출물)으로 학습됐다.
obstacles/ 1046장으로 확인한 결과, 주행 루프의 frame 을 그대로 넣었을 때
평균 conf 0.781 / conf>0.7 이 74% 인 반면 채널을 뒤집으면 0.516 / 28% 로 떨어진다.
raspi/L_7_YOLO.py 는 스왑을 안 하지만 그건 다른 모델(raspi/best.pt) 기준이다.

imgsz 를 낮추면 빨라지는 대신 탐지를 잃는다 (150장, 640 결과 기준):

    640 (학습값)  검출 117개          Mac(MPS) 31.7ms
    448           검출 105개 (90%)    18.2ms
    320           검출  85개 (73%)    11.2ms

ultralytics 는 Detector 를 만들 때 처음 import 한다 — 미설치 환경에서도
--yolo 를 안 주면 주행이 그대로 돌아야 하기 때문이다 (webui 가 Flask 를
지연 import 하는 것과 같은 이유).
"""
import collections
import os

import cv2
import numpy as np


class Detector:
    """YOLO 추론기. 마지막 결과를 들고 있는다."""

    def __init__(self, model_path, conf, imgsz):
        if not os.path.exists(model_path):
            raise SystemExit(
                f'모델 파일이 없습니다: {model_path}\n'
                '  .pt 는 git 에 없으므로 Pi 로 따로 복사해야 합니다.\n'
                '  경로를 직접 주려면: --yolo-model <경로>'
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit(
                'ultralytics 가 없어 객체 탐지를 켤 수 없습니다.\n'
                '  설치: pip install ultralytics\n'
                '  탐지 없이 주행하려면 --yolo 를 빼세요.'
            ) from exc

        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(model_path)
        self.names = self.model.names

        self.summary = '-'          # 'human 1, red 1' 형태. 화면에 그대로 나간다
        self.counts = {}            # {'human': 1, 'red': 1}
        self.boxes = []             # [(x1, y1, x2, y2, 이름, conf), ...] 원본 프레임 좌표
        self.total = 0
        self.runs = 0               # 추론 호출 횟수 — 주기가 맞는지 확인용

        # 첫 추론은 초기화 때문에 이후보다 훨씬 느리다 (Mac 에서 482ms vs 30ms).
        # 주행 중에 그 멈춤이 생기지 않도록 여기서 미리 한 번 돌려 둔다.
        self.infer(np.zeros((imgsz, imgsz, 3), dtype=np.uint8))
        self.summary, self.counts, self.total, self.runs = '-', {}, 0, 0
        self.boxes = []

    def infer(self, frame):
        """한 프레임 추론하고 요약 문자열을 돌려준다.

        frame 은 주행 루프의 것을 그대로 넣는다 (채널 변환 금지 — 모듈 docstring).
        """
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                 verbose=False, save=False)[0]
        # numpy 스칼라가 그대로 새어 나가면 /api/status 의 jsonify 가 500 을 낸다
        counts = collections.Counter(self.names[int(c)] for c in res.boxes.cls)

        self.boxes = [
            (int(b[0]), int(b[1]), int(b[2]), int(b[3]),
             self.names[int(c)], float(f))
            for b, c, f in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf)
        ]
        self.counts = {k: int(v) for k, v in counts.items()}
        self.total = int(sum(counts.values()))
        self.summary = ', '.join(f'{k} {v}' for k, v in sorted(self.counts.items())) or '-'
        self.runs += 1
        return self.summary


# 클래스별 고정 색상 (BGR). 같은 클래스가 항상 같은 색으로 보여야 눈이 익는다.
_COLORS = [(56, 56, 255), (10, 249, 72), (255, 143, 0), (49, 210, 207),
           (200, 0, 150), (23, 204, 146), (255, 38, 0)]


def _color(name):
    """클래스 이름 -> 고정 색상.

    hash() 는 문자열에 대해 실행마다 값이 달라진다(PYTHONHASHSEED). 그러면
    같은 클래스가 실행할 때마다 다른 색으로 보이므로 바이트 합을 쓴다.
    """
    return _COLORS[sum(name.encode()) % len(_COLORS)]


def draw_boxes(frame, boxes):
    """탐지 박스를 원본 프레임 좌표계에 그린다. 새 이미지를 돌려준다.

    cv2.putText 는 Hershey 폰트라 ASCII 만 나온다 — 클래스 이름이 전부
    영문이라 문제없다.
    """
    vis = frame.copy()
    h, w = vis.shape[:2]
    placed = []                     # 이미 놓은 라벨 사각형 [(x1,y1,x2,y2), ...]

    # 박스를 먼저 다 그린다 — 나중 라벨이 앞선 박스 선에 가려지지 않게
    for x1, y1, x2, y2, name, _ in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), _color(name), 2)

    for x1, y1, x2, y2, name, conf in boxes:
        label = f'{name} {conf:.2f}'
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lw, lh = tw + 4, th + base

        # 기본 위치는 박스 바로 위. 화면 위로 넘치면 박스 안쪽으로 내린다
        lx = max(0, min(x1, w - lw))
        ly = y1 - lh - 2 if y1 - lh - 2 >= 0 else y1 + 2

        # 앞서 놓은 라벨과 겹치면 한 줄씩 내린다. 겹친 객체가 여러 개일 때
        # 라벨이 서로 덮여 아무것도 못 읽는 것을 막는다.
        for _ in range(8):
            rect = (lx, ly, lx + lw, ly + lh)
            if not any(rect[0] < p[2] and p[0] < rect[2] and
                       rect[1] < p[3] and p[1] < rect[3] for p in placed):
                break
            ly += lh + 2
            if ly + lh > h:                     # 아래로도 넘치면 그냥 둔다
                break
        placed.append((lx, ly, lx + lw, ly + lh))

        cv2.rectangle(vis, (lx, ly), (lx + lw, ly + lh), _color(name), -1)
        cv2.putText(vis, label, (lx + 2, ly + th), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis
