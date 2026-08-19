# 미션 진입 면적(MISSION_AREA_ENTER) 측정 인터랙티브 뷰어

폴더 안에 있는 여러 이미지들을 **키보드 방향키(← / →)**로 앞뒤로 넘겨가며, 이미지 내 탐지된 객체의 **바운딩 박스 크기(Width, Height) 및 면적(Area = W × H)**을 실시간으로 확인하는 대화형 도구입니다.

결과 이미지를 파일로 저장하지 않고 화면에 즉시 띄워 빠르게 넘겨볼 수 있으며, 이전에 확인한 이미지는 캐싱되어 지연 없이 바로 탐색할 수 있습니다.

---

## 1. 파일 구조
```
temp-project/
├── mission_area/
│   ├── check_object_area.py   # 인터랙티브 방향키 뷰어 스크립트
│   └── README.md              # 사용 가이드
├── object_detection/
│   └── best_v6.pt             # YOLO 가중치 모델
└── lane-drive/
    └── config.py              # MISSION_AREA_ENTER 설정 파일
```

---

## 2. 사용 방법

### 방법 A: 코드 상단 변수 수정 후 실행
[`check_object_area.py`](file:///c:/project/temp-project/mission_area/check_object_area.py) 파일 상단의 `[사용자 설정 영역]`에서 `IMAGE_DIR`을 원하는 폴더 경로로 설정한 뒤 실행합니다.

```bash
python mission_area/check_object_area.py
```

### 방법 B: 터미널 명령어로 폴더 경로 직접 지정
```bash
# 특정 이미지 폴더 지정
python mission_area/check_object_area.py --image object_detection/today_photos

# 다른 모델 가중치 또는 임계값 지정
python mission_area/check_object_area.py --image object_detection/cars --conf 0.3
```

---

## 3. 키보드 조작 안내

| 기능 | 키 입력 |
|---|---|
| **다음 이미지** | **오른쪽 방향키 (`→`)**, `D`, `N`, `Space`, `Enter` |
| **이전 이미지** | **왼쪽 방향키 (`←`)**, `A`, `P`, `Backspace` |
| **맨 처음 이미지로** | `Home` |
| **맨 마지막 이미지로** | `End` |
| **종료** | **`Q`**, **`ESC`** |

---

## 4. 화면 및 콘솔 표시 내용
- **화면(GUI)**: 
  - 상단에 현재 이미지 번호와 파일명, 조작 키 안내 배너 표시
  - 바운딩 박스 위에 `클래스명 Area:면적 (너비x높이)` 형태(예: `car_white Area:2,436 (58x42)`)로 크기 표시
- **터미널 콘솔**:
  - 현재 이미지의 모든 객체 목록, Conf, Width, Height, Area(px), 그리고 [`lane-drive/config.py`](file:///c:/project/temp-project/lane-drive/config.py)의 `MISSION_AREA_ENTER` 기준치 대비 진입 여부 출력
