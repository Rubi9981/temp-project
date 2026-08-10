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
