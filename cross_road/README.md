# 교차로(Crossroad) 직진 및 객체 탐지 통합 자율주행 시스템

검정색 도로와 흰색 차선으로 이루어진 트랙에서 교차로를 만나 차선이 끊기거나 양옆으로 갈라질 때, 객체가 없으면 **`speed=50`과 직진 조향각**으로 안전하게 교차로를 통과하는 자율주행 프로그램입니다.

---

## 1. 주요 기능 및 알고리즘

1. **교차로 직진 주행 (Crossroad Straight Drive)**:
   - 전방 차선이 보이지 않거나 끊어짐 (`ctrl.ok == False`)
   - 화면에 7대 대상 객체(`red`, `left`, `right`, `car_red`, `car_white`, `human`, `right_sign`)가 미탐지됨
   - **동작**: 서보 모터를 직진 각도(`SERVO_CENTER = 90`)로 유지하며 천천히 교차로를 직진 통과 (`SPEED_CROSSROAD = 50`).
2. **일반 차선 추종 (Lane Following)**:
   - 차선이 정상 검출되면 Pure Pursuit 기하 제어 알고리즘으로 곡률을 추종하며 부드럽게 주행 (`SPEED_NORMAL = 100`).
3. **객체 감지 반응 및 안전 정지 (Safety Stop)**:
   - 적색 신호(`red`), 보행자(`human`), 차량(`car_red`, `car_white`) 탐지 시 즉시 모터를 세움 (`speed=0`).

---

## 2. 코드 상단 파라미터 튜닝 안내

`cross_road/drive_crossroad.py` 파일 최상단에서 모든 설정 변수를 한눈에 확인하고 바로 수정할 수 있습니다:

```python
# ==============================================================================
# ★★★ [사용자 튜닝 파라미터 / 전체 설정 변수] ★★★
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
SERVO_EMA_ALPHA        = 0.5         # 서보 평활 계수 (0.1 ~ 1.0)
INVERT_SERVO           = False       # 서보 방향 반전 여부

# 3. Pure Pursuit 자율주행 제어기 파라미터
LOOKAHEAD_CM           = 20.0        # 전방 주시 거리 (Lookahead distance, cm)
MAX_STEER_DEG          = 28.0        # 최대 앞바퀴 조향 각도 (deg)

# 4. 객체 탐지 (YOLO) 설정
TARGET_CLASSES         = ['red', 'left', 'right', 'car_red', 'car_white', 'human', 'right_sign']
SAFETY_STOP_CLASSES    = ['red', 'human', 'car_red', 'car_white']
YOLO_MODEL_PATH        = '../object_detection/best_v3_ncnn_model'
YOLO_CONF              = 0.25        # 신뢰도 임계값
YOLO_EVERY             = 3           # N프레임마다 YOLO 추론
```

---

## 3. 실행 방법

### (1) 라즈베리파이 실차 주행
```bash
cd cross_road
python3 drive_crossroad.py
```
- 브라우저에서 `http://<라즈베리파이 IP>:5000` 접속
- 안전을 위해 `STOP` 모드로 시작되므로 웹 화면에서 **[AUTO]** 버튼을 누르면 자율주행이 시작됩니다.

### (2) 모터 구동 없는 안전 테스트 (Dry-run)
```bash
python3 drive_crossroad.py --dry-run
```

### (3) 개발 PC에서 저장된 이미지로 검증 (Replay 모드)
```bash
python3 drive_crossroad.py --replay ../project/captures --replay-loop
```

---

## 4. 파일 구성

- [drive_crossroad.py](file:///c:/project/temp-project/cross_road/drive_crossroad.py): 메인 진입점 스크립트 (상단에 전체 파라미터 배치)
- [crossroad_driver.py](file:///c:/project/temp-project/cross_road/crossroad_driver.py): 차선 및 객체 상태 기반 주행 의사결정 상태 머신
- [config.py](file:///c:/project/temp-project/cross_road/config.py): 공통 전역 설정 및 차량 제원/BEV 좌표 관리
