# project_v2 — 차선 검출 파이프라인 재구성 및 평가 하네스

기존 `project/` 는 그대로 두고, 이진화·검출 방식을 **교체 가능한 백엔드**로 분리해
개선 효과를 숫자로 비교할 수 있게 만든 것.

## 구성

| 파일 | 역할 |
|---|---|
| `config.py` | 모든 파라미터의 단일 소스 (`SRC_POINTS` 포함). 기존엔 6개 파일에 복붙돼 있었다 |
| `bev.py` | BEV 변환 + 폭-기울기 품질 지표 |
| `binarize.py` | 이진화 백엔드 3종 — `hsv`(baseline) / `adaptive` / `tophat` |
| `detect.py` | 검출 백엔드 2종 — `centroid`(baseline) / `sliding` |
| `evaluate.py` | 데이터셋 전체 A/B 벤치마크 |
| `tune_src.py` | 폭-기울기 지표로 `SRC_POINTS` 최적화 (**아래 3번 항목 반드시 읽을 것**) |
| `label_gt.py` | 정답 라벨 클릭 도구 (선택) |
| `review.py` | 프레임별 시각 검토 뷰어 |
| `control.py` | Pure Pursuit 조향 제어기 |
| `calibrate_metric.py` | BEV 픽셀 <-> cm 환산 캘리브레이션 |
| `test_control.py` | Pure Pursuit 기하 단위 테스트 (40개) |
| `drive.py` | 라즈베리파이 실시간 주행 루프 (afb1) |

## 사용법

```bash
python evaluate.py --compare-all --src captures_bev   # 전체 조합 비교
python evaluate.py --binarize tophat --detect sliding  # 단일 조합 상세
python review.py                                       # 전체 순회 시각 검토
python review.py --sort worst                          # 오차 큰 순 — 실패 사례부터
python label_gt.py                                     # 정답 라벨링
python tune_src.py                                     # BEV 재튜닝 (dry-run)
```

`--src captures` 는 원본에서 매번 warp 하므로 `SRC_POINTS` 변경이 즉시 반영된다.
`--src captures_bev` 는 기존 BEV 산출물(68장)을 그대로 쓴다 — baseline 재현용.

---

## 측정 결과 (`captures_bev`, 68장)

```
binarize  detect         정상     단측     실패     과검출    폭std  err std
hsv       centroid      71%    18%    12%     26%   129.1     89.0   <- baseline
hsv       sliding       62%    24%    15%     26%    72.3     55.7
adaptive  centroid      91%     0%     9%     85%   105.2     36.8
adaptive  sliding       81%     9%    10%     85%    30.1     55.2
tophat    centroid      91%     4%     4%     37%    61.8     38.2
tophat    sliding       81%    10%     9%     37%    45.8     54.7
```

확실하게 말할 수 있는 것:

- **`tophat` 이 이진화로서 baseline보다 명확히 낫다.** 완전 실패 12% → 4%,
  단측(좌우 붕괴) 18% → 4%. 시각 확인 결과 baseline HSV는 정상 프레임에서도
  차선을 5조각으로 쪼개는 반면 tophat은 온전한 띠로 잡는다.
- **조향 포화가 사라졌다.** baseline은 `clip(±200)` 포화가 4장(6%) — 조향이
  풀락으로 나가는 프레임이다. baseline 외 모든 조합에서 0%.
- **`sliding` 이 좌우 붕괴를 실제로 복구한다.** 반사광 프레임
  `bev_frame_20260803_162326` 에서 baseline은 `center=515.9`(error −196, 풀락)로
  붕괴하지만 sliding은 `center=303.6` 으로 회복한다.
- **`adaptive` 는 수치가 좋아 보여도 쓰지 말 것.** 과검출 85%. 시각 확인 결과
  차선을 채우는 게 아니라 **차선의 윤곽선**을 뽑고 있다. 폭std가 낮은 것도
  윤곽 간 거리가 반복적이기 때문이지 차선을 잘 잡아서가 아니다.

확정할 수 없는 것:

- **`tophat+centroid` 와 `tophat+sliding` 중 무엇이 나은지는 지금 데이터로
  결정 불가.** err std는 centroid(38.2)가, 폭std는 sliding(45.8)이 낫다.
  정답 라벨이 없어 어느 쪽이 진짜 차선에 가까운지 판정할 수단이 없다.
  → `label_gt.py` 를 돌리면 `evaluate.py --gt` 가 실제 MAE/p95를 낸다. 5분이면 된다.

## 중심선 (`fit_center`)

`sliding` 검출기는 좌/우 차선을 2차 다항식 `x = f(y)` 로 피팅한다. `np.polyval` 은
계수에 선형이므로 **두 계수를 평균내면 그대로 중심선**이 된다.

```python
res = detect.sliding_window(mask)
x = np.polyval(res.fit_center, y)     # ROI 내 임의의 y 에서 주행 목표 x
```

한쪽 차선만 잡히면 상수항만 `LANE_WIDTH_PX/2` 만큼 밀어 평행 이동시킨다.
BEV에서 차선이 수직에 가까울 때만 정확한 근사이며, 곡률이 큰 구간에서는
법선 방향 간격이 이보다 좁아진다.

`centroid` 는 점 하나만 내므로 `fit_center` 가 `None` 이다 — 중심선을 보려면
반드시 `--detect sliding` 이어야 한다.

look-ahead 거리를 y로 지정해 목표점을 뽑을 수 있으므로 Pure Pursuit 같은
기하 제어기에 바로 물릴 수 있다.

## Pure Pursuit (`control.py`)

중심선 위에서 뒤축으로부터 **방사 거리**가 `L_d` 인 점을 찾아 원호로 통과한다.

```
alpha = atan2(Y, X)              목표점 방위각 (뒤축 기준)
kappa = 2Y / L_d^2               = 2 sin(alpha)/L_d, 삼각함수 없이 같은 값
delta = atan(L * kappa)          자전거 모델 조향각
servo = 90 - delta_deg * SERVO_PER_DEG   -> clip(30, 150)
```

부호는 **좌선회가 +**. 서보는 왼쪽이 작은 값이므로(`raspi/L_5_Capture.py`:
ArrowLeft -> 40) 마지막에 뒤집힌다.

### 실측값

| 항목 | 값 | 출처 |
|---|---|---|
| 차선 실폭 | 20 cm | 실측 |
| 휠베이스 | 11 cm | 실측 |
| `px_per_cm_x` | 22.875 | 457.5px(라벨 51쌍) / 20cm |
| `px_per_cm_y` | **12.0 (추정)** | 미실측 — `calibrate_metric.py` 필요 |
| `rear_axle_offset_cm` | **12.0 (추정)** | 미실측 |
| `LOOKAHEAD_CM` | 20 | 휠베이스의 1.5~3배(17~33cm) 중간값 |

BEV 가로 640px = 실제 28.0cm 다. **종방향 스케일과 뒤축 오프셋이 아직
추정값이라 servo 절대값은 신뢰하면 안 된다** — 상대적 경향만 본다.
`metric.json` 이 없으면 실행 시 경고가 뜬다.

### 좌표계

`bev.bev_to_vehicle()` / `vehicle_to_bev()`. 원점 = 뒤축 중심, **X 전방(+),
Y 좌측(+)**. BEV는 아래로 갈수록 차에 가까우므로 X 는 `(H-1 - y)` 에 비례한다.

x와 y의 cm 환산 계수가 다른 이유는 dst 사각형을 640×480 전체로 잡아 BEV를
만들었기 때문이다 (실세계 영역의 가로:세로 비와 무관하게 4:3 으로 강제).
현재 비등방 비율은 약 1.9 다.

## 실시간 주행 (`drive.py`)

라즈베리파이에서만 돈다. afb1 API 는 `raspi/` 예제에서 확인된 것만 쓴다:
`gpio.init/stby/servo/motor/stop_all`, `camera.init/get_image/release_camera`,
`flask.imshow`.

```bash
python3 drive.py --replay ../project/captures   # 0) 하드웨어 없이 임포트/로직 확인
python3 drive.py --dry-run --preview            # 1) 모터 끈 채 조향만 확인
python3 drive.py --speed 60 --record run1       # 2) 실제 주행 + 프레임 녹화
python3 review.py --src run1 --sort worst       # 3) 주행분을 오프라인에서 정밀 검토
```

**기본은 모터 정지다.** `--speed` 를 명시해야 움직인다.
서보가 반대로 돌면 `--invert-servo`.

### 디버그 창을 따로 만들지 않은 이유

`--preview` 가 `afb1.flask.imshow` 로 최소 오버레이(중심선/목표점/servo/fps)를
쏜다. 그 이상은 만들지 않았다.

30fps 로 흘러가는 화면으로는 "왜 저 프레임에서 틀렸는지"를 볼 수 없다.
**`--record` 로 저장해 두고 `review.py` 로 한 장씩 넘겨보는 쪽이 훨씬 낫다** —
좌우 다항식, 슬라이딩 윈도우, 정답 라벨, Pure Pursuit 원호가 전부 겹쳐 나오고
`--sort worst` 로 실패 프레임부터 볼 수 있다. 매 프레임 오버레이를 그리는
비용도 제어 루프가 쓸 CPU 다.

녹화 파일은 `captures/` 와 같은 형식(채널 스왑 후 640×480)이라 `review.py`
`evaluate.py` 가 그대로 처리한다. 덤으로 **시간축 추적에 필요한 연속 주행
데이터가 이때 확보된다** — 지금까지 계속 막혀 있던 지점이다.

### 채널 스왑 주의

`raspi/L_5_Capture.py` 는 저장 전에 `cv2.cvtColor(frame, COLOR_BGR2RGB)` 를
거친 뒤 `cv2.imwrite` 한다. 따라서 `captures/` 를 `cv2.imread` 로 읽은 배열은
**`get_image()` 원본이 아니라 채널이 한 번 뒤바뀐 것**이다. 오프라인에서
튜닝한 이진화가 같은 색을 보려면 실시간 루프도 똑같이 스왑해야 하며,
`Afb1Hardware.read()` 가 그렇게 한다. 이걸 빠뜨리면 특히 `hsv` 백엔드가
전혀 다른 색을 보게 된다.

### 검출 실패 처리

정적 평가에서 중심선 미산출이 74장 중 9장(12%)이었다. 30fps 면 초당 서너
프레임은 실패한다는 뜻이라 정의된 동작이 필요하다.

```
실패 1~4프레임  : 직전 서보 명령 유지하고 계속 주행
실패 5프레임 이상: 모터 정지 (조향은 직전 값 유지)
종료/예외/Ctrl+C : finally 에서 motor(0) + servo(90) + stop_all()
```

서보 명령에는 지수이동평균(`SERVO_EMA_ALPHA=0.5`)을 건다. 끄려면 `--ema 1.0`.

### 성능

Mac 기준 프레임당 `warp 0.55ms + adaptive 0.50ms + sliding 1.32ms = 약 2.4ms`.
Pi 는 5~10배 느리다고 보면 15~25ms 로 30fps 는 무리 없을 전망이다.
**다만 이건 추정이며 실측이 필요하다** — `drive.py` 가 fps 를 찍어준다.
`tophat` 은 Mac 에서 0.7ms 더 든다.

## 라벨링 (`label_gt.py`)

**차선 띠의 가운데를 클릭한다.** 검출기가 추정하는 값이 띠의 중심이기 때문이다
(centroid는 컨투어 무게중심, sliding은 차선 픽셀 평균 x). 라벨 행 y=372에서
띠 폭은 중앙값 37px 이라 경계선을 찍으면 18px쯤 치우친다.

라벨 행에서의 실제 분포:

```
좌우 둘 다 보임 : 54장   -> 클릭 2번
한쪽만 보임     :  8장   -> 클릭 1번 + l (좌측) 또는 r (우측)
아무것도 없음   :  6장   -> s
```

한쪽만 보이는 8장이 **baseline이 좌우 붕괴로 풀락 조향을 내던 프레임**이라
반드시 `l`/`r` 로 어느 쪽인지 지정해야 한다. `s` 로 넘기면 채점에서 빠진다.

`evaluate.py --gt` 는 둘을 나눠 집계한다:

- **GT중심** — 좌우 둘 다 보이는 프레임의 차선 중심 오차
- **GT단측** — 한쪽만 보이는 프레임의 **해당 차선 위치** 오차

단측 프레임의 "중심"은 채점하지 않는다. 정답 중심을 알려면 차선 폭을 가정해야
하는데, 그건 검출기가 외삽에 쓰는 값과 같아 순환 논증이 된다.

---

## 작업 중 확인된 세 가지

### 1. baseline 수치 정정

앞선 대화에서 보고한 "정상 검출 38%, 과검출 32%"는 임시 측정 스크립트의 실수였다.
`cv2.morphologyEx(m, op, k, 1)` 의 4번째 위치인자는 `iterations` 가 아니라 `dst` 라서
CLOSE가 2회가 아닌 1회만 돌았다.

`lane_detector_v2.py:100-102` 의 실제 동작(`iterations=` 키워드) 기준 정확한 baseline:

```
0개 = 12%   1개 = 18%   정확히 2개 = 44%   3개 이상 = 26%
```

실패율(12%)·단측(18%)·error std(89.0)는 영향이 없었다.

### 2. `tune_src.py` 의 최적화는 검증을 통과하지 못했다 — 적용하지 않았다

폭-기울기 지표 자체는 개선됐다:

```
median slope  +0.391 -> +0.139      (직선 프레임 mean|slope| 0.458 -> 0.202)
```

그런데 그 `SRC_POINTS` 를 실제로 적용하고 `evaluate.py` 를 돌리자 **하류 지표가
전면 후퇴**했다:

```
폭std (sliding 기준)   hsv 71.3 -> 116.0    adaptive 39.0 -> 67.0    tophat 50.1 -> 74.1
```

프록시(프레임 **내부** 폭 기울기)를 최적화한 결과가 실제 목표(프레임 **간** 폭
일관성)를 악화시킨, 전형적인 프록시 게이밍이다. 최적화기가 사다리꼴의 세로 폭을
145px → 102px 로 줄여 수직 확대율을 키운 탓이 크다.

**따라서 `calib.json` 을 생성하지 않았고 기본 `SRC_POINTS` 가 그대로 쓰인다.**
`tune_src.py` 는 도구로서 남겨두되, 지금 상태로 `--save` 하는 것은 권하지 않는다.

잔차 +0.139가 남는 구조적 이유가 있다. **호모그래피는 배럴 왜곡으로 휜 선을
펼 수 없다.** 원본 `1go.jpg` 의 벽/바닥 경계선(실세계 직선)에서 이미지 중간
반경 기준 4~5px 의 휨이 측정됐고, 방사 왜곡은 r³에 비례하므로 가장자리에서는
더 커진다. 체커보드 캘리브레이션으로 왜곡을 먼저 펴지 않는 한 이 잔차는
기하 파라미터를 어떻게 맞춰도 남는다.

### 3. 프록시 지표의 한계

정답 라벨이 없으면 "반사광 두 개를 차선으로 오인"한 경우도 `정상 검출`로 집계된다.
차선 폭 표준편차가 그 방어막이지만 완전하지 않다 — `adaptive` 가 그 예다
(폭std 최저이면서 실제로는 윤곽선을 잡고 있음).

`evaluate.py` 리포트는 이 한계를 매번 출력한다. 순위를 확정하려면 `label_gt.py`.

---

## 범위 밖 (다음 작업)

- **시간축 추적 / 게이팅 / 동적 ROI** — 효과가 가장 클 것으로 보이나 **연속 프레임
  영상이 없어 검증이 성립하지 않는다.** 74장은 서로 이어지지 않는 단일 이미지다.
  실패 프레임 `bev_frame_20260803_162233` 은 급커브에서 차선이 ROI를 완전히
  벗어난 경우로, **어떤 이진화 방법으로도 못 고친다**(세 백엔드 모두 blobs=0 확인).
  동적 ROI가 필요하고, 그러려면 주행 영상이 있어야 한다.
- **편광 필터 / 노출 고정** — 하드웨어·펌웨어 측. `raspi/` 의 `afb1.camera.init`.
  대회규정상 개조 허용 범위 확인 필요
- **체커보드 캘리브레이션 / 파라메트릭 BEV** — 위 2번에서 보듯 BEV 기하 개선의
  선행 조건. `config.py` 의 `CAMERA_MATRIX` / `DIST_COEFFS` 자리를 비워뒀다
- **CNN 세그멘테이션** — 라벨 데이터 부재

## 기존 `project/` 에서 발견된 문제 (수정하지 않고 기록만)

- `lane_detector_v2.py:11`, `bev.py:6`, `measure_scale.py:10` 이 참조하는
  `frame_20260803_162053.jpg`, `frame_20260803_162123.jpg` 가 **존재하지 않는다.**
  `SRC_POINTS` 를 뽑은 원본이라 재검증이 불가능하다
- `classify_captures.py:136` 이 `shutil.move` 로 원본을 하위 폴더로 **이동**시킨다.
  실행하면 `captures/` 가 재편되어 이 데이터셋이 깨진다 (현재는 미실행 상태)
- `lane_detector_v2.py:75,78` 의 `matrix_inv` 와 `project_point()` 가 정의만 되고
  미사용 — `review.py` 에서 역투영 시각화로 살렸다
