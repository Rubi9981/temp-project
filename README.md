# Lane Tracer

라즈베리파이 기반 자율주행 차량의 차선 인식, 조향 제어, 객체 탐지 코드를 관리하는 프로젝트입니다.

## 디렉토리 구조

```text
.
├── lane-drive/          # 실시간 차선 주행 및 객체 탐지 코드
│   ├── drive.py         # 자율주행 진입점
│   ├── watch.py         # 카메라 및 객체 탐지 모니터링
│   ├── yolo_server.py   # 노트북에서 실행하는 원격 YOLO 서버
│   ├── bev.py           # Bird's Eye View 변환
│   ├── binarize.py      # 차선 후보 이진화
│   ├── detect.py        # 차선 검출
│   ├── control.py       # 조향 제어
│   ├── driver.py        # 주행 판단 로직
│   └── hardware.py      # 카메라·모터·서보 하드웨어 제어
├── object_detection/    # YOLO 모델 파일
├── project/             # 차선 검출 관련 실험 및 도구
├── raspi/               # 라즈베리파이용 기존 예제 코드
├── lecture/             # OpenCV·AI 학습 예제
├── docs/                # 프로젝트 문서
├── ARCHITECTURE.md      # 전체 구조 및 동작 설명
└── DEVLOG.md            # 개발 기록
```

## 코드 실행 방법

노트북과 라즈베리파이는 같은 네트워크에 연결되어 있어야 합니다. 먼저 노트북에서 YOLO 서버를 실행한 뒤, 라즈베리파이에서 주행 코드를 실행합니다.

### 노트북에서 YOLO 서버 실행

프로젝트 루트 디렉토리에서 실행합니다.

```bash
python3 lane-drive/yolo_server.py \
  --host 0.0.0.0 \
  --port 5010
```

### 라즈베리파이에서 주행 코드 실행

라즈베리파이에서 프로젝트 디렉토리로 이동한 후 실행합니다.

```bash
cd ~/afb_home/temp-project
```

```bash
python3 lane-drive/drive.py \
  --profile \
  --log-every 10 \
  --speed 150 \
  --pace-fps 0 \
  --steer-gain 5.0 \
  --yolo-remote 192.168.4.19:5010
```

### 라즈베리파이에서 탐지 화면 확인

```bash
python3 lane-drive/watch.py --yolo-remote 192.168.4.19:5010
```

`192.168.4.19`는 YOLO 서버를 실행한 노트북의 IP 주소입니다. 노트북의 IP가 다르면 두 명령어의 주소를 실제 노트북 IP로 변경해야 합니다.
