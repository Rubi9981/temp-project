"""미션 진입 크기(MISSION_AREA_ENTER) 측정을 위한 대화형(Interactive) 객체 탐지 뷰어.

폴더 내 이미지들을 키보드 방향키(← / →)로 앞뒤로 넘겨가며
탐지된 객체의 바운딩 박스, 클래스명, 크기(Area, 너비x높이)를 실시간으로 확인할 수 있습니다.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ==============================================================================
# [사용자 설정 영역] - 여기서 모든 주요 변수와 경로를 확인 및 변경할 수 있습니다.
# ==============================================================================

# 1. 탐지할 이미지가 들어있는 폴더 경로 (또는 단일 파일 경로)
#    - 예시: "object_detection/cars"
#    - 예시: "object_detection/today_photos"
IMAGE_DIR = "object_detection/traffic_lights_2"
#IMAGE_DIR = "mission_area/right_sign"


# 2. YOLO 가중치 모델 파일 경로
MODEL_PATH = "object_detection/best_v6.pt"

# 3. YOLO 추론 파라미터
CONF_THRESHOLD = 0.25   # 탐지 신뢰도 임계값 (0.0 ~ 1.0)
IMG_SIZE = 640          # YOLO 입력 이미지 크기

# 4. 시각화 그래픽 설정
FONT_SCALE = 0.5        # 라벨 텍스트 글자 크기
BOX_THICKNESS = 2       # 바운딩 박스 테두리 두께

# 5. 현재 config.py에 설정된 MISSION_AREA_ENTER 기준값 (비교용)
#    - 탐지된 객체의 Area와 비교하여 터미널에 상태를 출력해줍니다.
MISSION_AREA_ENTER_CURRENT = {
    'right_sign': 2500,
    'red': 1100,
    'left': 500,
    'right': 1000,
    'human': 11000,
    'car_red': 2000,
    'car_white': 3300,
}

# ==============================================================================


# 클래스별 고정 색상 팔레트 (BGR 형식)
CLASS_COLORS = {
    'red': (0, 0, 255),           # 빨간색
    'left': (255, 200, 0),        # 하늘색 계열
    'right': (0, 255, 255),       # 노란색
    'right_sign': (255, 140, 0),  # 주황 계열
    'human': (0, 255, 0),         # 초록색
    'car_red': (70, 70, 220),     # 진빨강
    'car_white': (220, 220, 220), # 밝은 회색
}

FALLBACK_COLORS = [
    (56, 56, 255), (10, 249, 72), (255, 143, 0),
    (49, 210, 207), (200, 0, 150), (23, 204, 146), (255, 38, 0)
]


def get_class_color(name: str):
    """클래스명에 따른 고유 색상 반환."""
    if name in CLASS_COLORS:
        return CLASS_COLORS[name]
    return FALLBACK_COLORS[sum(name.encode('utf-8')) % len(FALLBACK_COLORS)]


def load_yolo_model(model_path: str):
    """YOLO 모델 로드."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[오류] YOLO 모델 파일을 찾을 수 없습니다: {model_path}\n"
            f"현재 작업 디렉토리: {os.getcwd()}"
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "[오류] ultralytics 패키지가 설치되어 있지 않습니다.\n"
            "설치 명령: pip install ultralytics"
        ) from exc

    print(f"[정보] YOLO 모델 로딩 중: {model_path}")
    model = YOLO(model_path)
    print(f"[정보] 모델 클래스 목록 ({len(model.names)}개): {model.names}\n")
    return model


def get_image_list(target_path: str):
    """지정된 경로에서 이미지 파일 목록을 수집."""
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"[오류] 지정한 이미지 경로를 찾을 수 없습니다: {target_path}")

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    if os.path.isfile(target_path):
        if target_path.lower().endswith(valid_exts):
            return [target_path]
        else:
            raise ValueError(f"[오류] 지원하지 않는 이미지 확장자입니다: {target_path}")

    # 디렉토리인 경우
    image_files = []
    for root, _, files in os.walk(target_path):
        for f in sorted(files):
            if f.lower().endswith(valid_exts):
                image_files.append(os.path.join(root, f))

    if not image_files:
        raise FileNotFoundError(f"[경고] 디렉토리 내에 이미지 파일이 없습니다: {target_path}")

    return sorted(image_files)


def draw_boxes_with_size(frame: np.ndarray, detections: list, info_header: str = ""):
    """탐지된 객체의 바운딩 박스와 [클래스명 | 크기(Area / WxH)]를 이미지에 그립니다."""
    vis = frame.copy()
    img_h, img_w = vis.shape[:2]
    placed_labels = []

    # 1. 상단 정보 배너 (현재 인덱스 및 키 안내)
    if info_header:
        cv2.rectangle(vis, (0, 0), (img_w, 28), (30, 30, 30), -1)
        cv2.putText(
            vis, info_header, (10, 19),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
        )

    # 2. 바운딩 박스 사각형 그리기
    for det in detections:
        x1, y1, x2, y2 = det['box']
        color = get_class_color(det['name'])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    # 3. 클래스명과 크기 라벨 그리기
    for det in detections:
        x1, y1, x2, y2 = det['box']
        name = det['name']
        conf = det['conf']
        w = det['w']
        h = det['h']
        area = det['area']
        color = get_class_color(name)

        # 라벨 텍스트: "클래스명  Area:면적 (너비x높이)"
        full_label = f"{name}  Area:{area:,} ({w}x{h})"

        # 텍스트 크기 계산
        (tw, th), base = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
        lw, lh = tw + 8, th + base + 6

        # 라벨 기본 위치 (박스 바로 위, 화면 위로 벗어나거나 배너에 걸리면 박스 내부로)
        min_y = 30 if info_header else 0
        lx = max(0, min(x1, img_w - lw))
        ly = y1 - lh - 2 if y1 - lh - 2 >= min_y else y1 + 2

        # 겹치는 라벨 위치 조정
        for _ in range(10):
            rect = (lx, ly, lx + lw, ly + lh)
            overlap = any(
                rect[0] < p[2] and p[0] < rect[2] and rect[1] < p[3] and p[1] < rect[3]
                for p in placed_labels
            )
            if not overlap:
                break
            ly += lh + 2
            if ly + lh > img_h:
                break

        placed_labels.append((lx, ly, lx + lw, ly + lh))

        # 라벨 배경 박스
        cv2.rectangle(vis, (lx, ly), (lx + lw, ly + lh), color, -1)

        # 텍스트 렌더링 (배경색 밝기에 따라 검정 또는 흰색 폰트)
        brightness = (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299)
        text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)

        cv2.putText(
            vis, full_label, (lx + 4, ly + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, text_color, 1, cv2.LINE_AA
        )

    return vis


def infer_single_image(model, img_path: str):
    """단일 이미지에 대해 YOLO 추론을 수행하고 결과 딕셔너리 리스트를 반환."""
    frame = cv2.imread(img_path)
    if frame is None:
        return None, []

    results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]

    detections = []
    for box, cls_idx, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls_idx = int(cls_idx)
        class_name = model.names[cls_idx]
        conf_val = float(conf)

        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = w * h

        detections.append({
            'name': class_name,
            'conf': conf_val,
            'box': (x1, y1, x2, y2),
            'w': w,
            'h': h,
            'area': area
        })

    return frame, detections


def print_detection_table(img_path: str, frame_shape: tuple, detections: list, current_idx: int, total_imgs: int):
    """콘솔에 탐지 결과 요약 테이블 출력."""
    print("\n" + "=" * 88)
    print(f" [{current_idx + 1}/{total_imgs}] {os.path.basename(img_path)} (해상도: {frame_shape[1]}x{frame_shape[0]})")
    print(f" 전체 경로: {img_path}")
    print(f" 탐지된 객체 수: {len(detections)}개")
    print("-" * 88)
    print(f"{'No':<3} | {'Class':<12} | {'Conf':<6} | {'Width':<6} | {'Height':<6} | {'Area (px)':<10} | {'현재 기준(config)':<16} | {'진입 여부'}")
    print("-" * 88)

    if not detections:
        print("   (탐지된 객체가 없습니다)")
    else:
        for idx, det in enumerate(detections, start=1):
            name = det['name']
            conf_str = f"{det['conf']:.2f}"
            w = det['w']
            h = det['h']
            area = det['area']

            cfg_threshold = MISSION_AREA_ENTER_CURRENT.get(name, None)
            if cfg_threshold is not None:
                cfg_str = f"{cfg_threshold:,} px"
                enter_status = "★ 진입(Area >= 기준)" if area >= cfg_threshold else "  미진입(작음)"
            else:
                cfg_str = "미지정"
                enter_status = "-"

            print(f"{idx:<3} | {name:<12} | {conf_str:<6} | {w:<6} | {h:<6} | {area:<10,}| {cfg_str:<16} | {enter_status}")

    print("=" * 88)
    print(" [조작키]  ← / A / P : 이전 이미지   |   → / D / N / Space : 다음 이미지   |   Q / ESC : 종료\n")


def parse_arguments():
    """커맨드라인 인자 파싱."""
    parser = argparse.ArgumentParser(description="미션 객체 탐지 및 크기 대화형 뷰어")
    parser.add_argument("--image", "-i", type=str, default=IMAGE_DIR,
                        help="분석할 폴더 또는 이미지 파일 경로 (기본값: 상단 IMAGE_DIR)")
    parser.add_argument("--model", "-m", type=str, default=MODEL_PATH,
                        help="YOLO 모델 가중치 파일 경로 (기본값: 상단 MODEL_PATH)")
    parser.add_argument("--conf", "-c", type=float, default=CONF_THRESHOLD,
                        help="신뢰도 임계값 (기본값: 상단 CONF_THRESHOLD)")
    return parser.parse_args()


def main():
    args = parse_arguments()

    img_target = args.image
    model_target = args.model

    print("\n" + "=" * 68)
    print(" [미션 객체 탐지 & 크기 측정 인터랙티브 뷰어 (방향키 탐색)] ")
    print("=" * 68)

    # 1. 모델 로드
    model = load_yolo_model(model_target)

    # 2. 이미지 파일 목록 가져오기
    image_paths = get_image_list(img_target)
    total_imgs = len(image_paths)
    print(f"[정보] 총 {total_imgs}개의 이미지를 불러왔습니다.")
    print("--------------------------------------------------------------------")
    print(" [키보드 안내]")
    print("   ▶ 다음 이미지 : 오른쪽 방향키(→)  또는  D / N / Space")
    print("   ◀ 이전 이미지 : 왼쪽 방향키(←)    또는  A / P / Backspace")
    print("   ⏹ 종료         : Q  또는  ESC")
    print("--------------------------------------------------------------------\n")

    # 빠른 탐색을 위한 결과 캐시 (index -> (vis_img, detections))
    cache = {}
    current_idx = 0
    window_name = "Mission Object Area Viewer (Use Left/Right Arrow Keys, Q to Quit)"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        img_path = image_paths[current_idx]

        # 캐시에 없으면 추론 수행
        if current_idx not in cache:
            frame, detections = infer_single_image(model, img_path)
            if frame is None:
                print(f"[경고] 이미지를 불러올 수 없습니다: {img_path}")
                # 다음 이미지로 자동 이동
                if current_idx < total_imgs - 1:
                    current_idx += 1
                    continue
                else:
                    break

            info_header = f"[{current_idx + 1}/{total_imgs}] {os.path.basename(img_path)}  |  [<-- / -->] Navigate  |  [Q] Quit"
            vis_img = draw_boxes_with_size(frame, detections, info_header=info_header)
            cache[current_idx] = (vis_img, detections, frame.shape)
        else:
            vis_img, detections, frame_shape = cache[current_idx]

        # 콘솔에 상세 표 출력
        frame_shape = cache[current_idx][2]
        print_detection_table(img_path, frame_shape, detections, current_idx, total_imgs)

        # 화면 표시
        cv2.imshow(window_name, vis_img)

        # 키 입력 대기 (방향키 및 다양한 단축키 지원)
        # cv2.waitKeyEx 는 특수키(방향키 등)의 OS별 전체 키코드를 반환합니다.
        key = cv2.waitKeyEx(0)

        # 1) 종료 키 (q, Q, ESC)
        if key in (ord('q'), ord('Q'), 27):
            print("\n[알림] 뷰어를 종료합니다.")
            break

        # 2) 다음 이미지 키:
        #    - Windows Right Arrow: 2555904 (0x270000) or 39
        #    - Linux Right Arrow: 65363
        #    - macOS Right Arrow: 63235
        #    - 영문 d, D, n, N, Spacebar (32), Enter (13)
        elif key in (2555904, 39, 65363, 63235, ord('d'), ord('D'), ord('n'), ord('N'), 32, 13):
            if current_idx < total_imgs - 1:
                current_idx += 1
            else:
                print(">> [마지막 이미지입니다]")

        # 3) 이전 이미지 키:
        #    - Windows Left Arrow: 2424832 (0x250000) or 37
        #    - Linux Left Arrow: 65361
        #    - macOS Left Arrow: 63234
        #    - 영문 a, A, p, P, Backspace (8)
        elif key in (2424832, 37, 65361, 63234, ord('a'), ord('A'), ord('p'), ord('P'), 8):
            if current_idx > 0:
                current_idx -= 1
            else:
                print("<< [첫 번째 이미지입니다]")

        # 4) 처음으로 이동 (Home: 2359296 / 36 / 65360)
        elif key in (2359296, 36, 65360):
            current_idx = 0
            print("<< [맨 처음 이미지로 이동했습니다]")

        # 5) 끝으로 이동 (End: 2293760 / 35 / 65367)
        elif key in (2293760, 35, 65367):
            current_idx = total_imgs - 1
            print(">> [맨 마지막 이미지로 이동했습니다]")

    cv2.destroyAllWindows()
    print("[완료] 프로그램이 종료되었습니다.\n")


if __name__ == '__main__':
    main()
