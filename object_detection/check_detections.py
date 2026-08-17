import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# ⚙️ [사용자 설정] 파일 및 폴더 경로 설정
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

# 1) YOLO 모델 파일 경로 (.pt)
MODEL_PATH = BASE_DIR / "best_v3.pt"

# 2) 탐지할 이미지가 들어있는 폴더 경로
PHOTOS_DIR = BASE_DIR / "today_photos"

# 3) 객체 탐지 신뢰도(Confidence) 임계값
CONF_THRESHOLD = 0.25
# ==============================================================================

# 클래스별 고유 색상 팔레트 (BGR)
COLOR_PALETTE = [
    (56, 56, 255),   # Red
    (10, 249, 72),   # Green
    (255, 143, 0),   # Blue
    (49, 210, 207),  # Yellow
    (200, 0, 150),   # Magenta/Purple
    (23, 204, 146),  # Cyan-Green
    (255, 38, 0),    # Deep Blue
    (134, 219, 61),  # Lime
    (187, 212, 0),   # Cyan
    (255, 194, 0),   # Sky Blue
    (168, 153, 44),  # Teal
    (151, 157, 255), # Coral
]

def get_color(cls_id):
    """클래스 ID에 대응하는 색상 반환"""
    return COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]

def is_overlapping(box1, box2):
    """두 사각형 [x1, y1, x2, y2] 간의 교차(겹침) 여부 확인"""
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

def draw_detections_no_overlap(image, boxes, names, top_margin=45):
    """
    바운딩 박스와 라벨을 그리되, 하나의 객체에 여러 박스/라벨이 검출되거나 
    인접한 위치의 라벨들이 서로 겹치지 않도록 스마트하게 스택/배치합니다.
    """
    img = image.copy()
    h, w = img.shape[:2]

    if boxes is None or len(boxes) == 0:
        return img

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thickness = 1

    # 1. 모든 바운딩 박스 정보 파싱 및 박스 그리기
    parsed_boxes = []
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
        conf = float(boxes.conf[i].cpu().numpy())
        cls_id = int(boxes.cls[i].cpu().numpy())
        cls_name = names.get(cls_id, str(cls_id))
        color = get_color(cls_id)
        
        x1, y1, x2, y2 = xyxy
        # 바운딩 박스 그리기 (선 두께 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        parsed_boxes.append({
            "xyxy": (x1, y1, x2, y2),
            "conf": conf,
            "cls_id": cls_id,
            "cls_name": cls_name,
            "color": color,
            "label": f"{cls_name} {conf:.2f}"
        })

    # y1 좌표 기준으로 정렬하여 상단부터 차례대로 라벨 배치
    parsed_boxes.sort(key=lambda b: (b["xyxy"][1], b["xyxy"][0]))

    # 2. 라벨 겹침 방지 배치 계산
    placed_label_rects = []  # [[lx1, ly1, lx2, ly2], ...]

    for item in parsed_boxes:
        x1, y1, x2, y2 = item["xyxy"]
        label = item["label"]
        color = item["color"]

        # 텍스트 크기 측정
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        lw = tw + 8
        lh = th + baseline + 6

        # 초기 위치: 박스 좌상단 바로 위
        lx1 = x1
        ly2 = y1 - 2
        ly1 = ly2 - lh

        # 상단 오버레이 바(top_margin)보다 위로 침범하면 박스 안쪽 상단으로 시작
        if ly1 < top_margin:
            ly1 = y1 + 2
            ly2 = ly1 + lh

        # 화면 좌우 경계 조정
        if lx1 + lw > w - 4:
            lx1 = max(4, w - lw - 4)
        if lx1 < 4:
            lx1 = 4
        lx2 = lx1 + lw

        cur_rect = [lx1, ly1, lx2, ly2]

        # 겹침 검사 및 회피 (충돌 시 위로 스택 -> 화면 밖이면 아래로 스택 -> 필요시 옆으로 이동)
        max_attempts = 20
        attempts = 0
        while attempts < max_attempts:
            collision = False
            for prev_rect in placed_label_rects:
                if is_overlapping(cur_rect, prev_rect):
                    collision = True
                    # 1순위: 이전 라벨 바로 위로 올리기
                    new_ly2 = prev_rect[1] - 2
                    new_ly1 = new_ly2 - lh
                    
                    # 위쪽 공간이 부족할 경우(상단 마진 침범): 이전 라벨 바로 아래로 내리기
                    if new_ly1 < top_margin:
                        new_ly1 = prev_rect[3] + 2
                        new_ly2 = new_ly1 + lh

                    # 아래쪽 공간도 부족할 경우: x축으로 오프셋
                    if new_ly2 > h - 4:
                        cur_rect[0] = prev_rect[2] + 4
                        cur_rect[2] = cur_rect[0] + lw
                        new_ly1 = max(top_margin + 2, y1 - lh - 2)
                        new_ly2 = new_ly1 + lh

                    cur_rect[1] = new_ly1
                    cur_rect[3] = new_ly2
                    break
            if not collision:
                break
            attempts += 1

        # 최종 화면 경계 재보정
        cur_rect[0] = max(2, min(cur_rect[0], w - lw - 2))
        cur_rect[2] = cur_rect[0] + lw
        cur_rect[1] = max(top_margin + 2, min(cur_rect[1], h - lh - 2))
        cur_rect[3] = cur_rect[1] + lh

        placed_label_rects.append(cur_rect)

        # 3. 라벨 배경 및 텍스트 렌더링
        flx1, fly1, flx2, fly2 = cur_rect

        # 라벨이 원래 박스 좌상단에서 멀어졌을 경우 연결선(Leader Line) 표시
        if abs(flx1 - x1) > 20 or abs(fly2 - y1) > 20:
            cv2.line(img, (x1, y1), (flx1, fly2), color, 1, cv2.LINE_AA)
            cv2.circle(img, (x1, y1), 3, color, -1)

        # 라벨 배경 채우기 (클래스별 색상)
        cv2.rectangle(img, (flx1, fly1), (flx2, fly2), color, -1)
        # 라벨 외곽선 (가독성 향상)
        cv2.rectangle(img, (flx1, fly1), (flx2, fly2), (20, 20, 20), 1)

        # 배경색 밝기에 따라 텍스트 색상 자동 선택 (검정/흰색)
        b, g, r = color
        luminance = (0.299 * r + 0.587 * g + 0.114 * b)
        text_color = (0, 0, 0) if luminance > 145 else (255, 255, 255)

        text_x = flx1 + 4
        text_y = fly1 + th + 3
        cv2.putText(img, label, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return img

def main():
    # 1. 파일 및 폴더 경로 유효성 검사
    if not MODEL_PATH.exists():
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return
    if not PHOTOS_DIR.exists():
        print(f"[오류] 이미지 폴더를 찾을 수 없습니다: {PHOTOS_DIR}")
        return

    # 2. 이미지 파일 목록 가져오기
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([f for f in PHOTOS_DIR.iterdir() if f.suffix.lower() in valid_extensions])

    if not image_files:
        print(f"[오류] '{PHOTOS_DIR}' 폴더에 이미지 파일이 없습니다.")
        return

    print("=" * 65)
    print(f" 🔍 YOLO 모델 탐지 뷰어")
    print(f" - 모델 파일   : {MODEL_PATH.name}")
    print(f" - 이미지 경로 : {PHOTOS_DIR.name} (총 {len(image_files)}장)")
    print(f" - 신뢰도 임계 : {CONF_THRESHOLD}")
    print("=" * 65)
    print("\n모델을 불러오는 중입니다...")
    
    # 3. YOLO 모델 로드
    model = YOLO(str(MODEL_PATH))

    window_name = "YOLO Detection Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    current_idx = 0
    num_images = len(image_files)

    print("\n[조작 가이드]")
    print(" - [Enter] / [Space]     : 다음 이미지로 이동")
    print(" - [Backspace] / [P 키]   : 이전 이미지로 이동")
    print(" - [Q 키] / [ESC 키]     : 프로그램 종료\n")

    while 0 <= current_idx < num_images:
        img_path = image_files[current_idx]

        # 원본 이미지 로드
        origin_img = cv2.imread(str(img_path))
        if origin_img is None:
            print(f"[경고] 이미지를 불러올 수 없습니다: {img_path.name}")
            current_idx += 1
            continue

        # 4. YOLO 객체 탐지 추론
        results = model.predict(source=origin_img, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]
        boxes = result.boxes

        # 5. 라벨 겹침 방지 커스텀 렌더링 적용
        annotated_img = draw_detections_no_overlap(
            image=origin_img,
            boxes=boxes,
            names=model.names,
            top_margin=42
        )

        # 6. 탐지된 객체 정보 요약 파악
        det_summary = {}
        if boxes is not None and len(boxes) > 0:
            for cls_id in boxes.cls:
                cls_name = model.names[int(cls_id)]
                det_summary[cls_name] = det_summary.get(cls_name, 0) + 1
        summary_str = ", ".join([f"{k}: {v}" for k, v in det_summary.items()]) if det_summary else "탐지된 객체 없음"

        # 콘솔 출력
        print(f"[{current_idx + 1}/{num_images}] {img_path.name}  ==>  {summary_str}")

        # 7. 상단 정보 오버레이 바 (배경 박스 + 텍스트)
        h, w = annotated_img.shape[:2]
        cv2.rectangle(annotated_img, (0, 0), (w, 40), (40, 40, 40), -1)
        info_text = f"[{current_idx + 1}/{num_images}] {img_path.name} | {summary_str}"
        cv2.putText(
            annotated_img,
            info_text,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        # 8. 화면에 출력
        cv2.imshow(window_name, annotated_img)

        # 9. 키 입력 대기
        key = cv2.waitKey(0) & 0xFF

        # [Enter] (13, 10), [Space] (32), [N 키] (ord('n'), ord('N'))
        if key in (13, 10, 32, ord('n'), ord('N')):
            current_idx += 1
        # [Backspace] (8), [P 키] (ord('p'), ord('P'))
        elif key in (8, ord('p'), ord('P')):
            current_idx = max(0, current_idx - 1)
        # [Q 키] (ord('q'), ord('Q')), [ESC] (27)
        elif key in (27, ord('q'), ord('Q')):
            print("\n사용자에 의해 프로그램을 종료합니다.")
            break

        # 마우스로 X 버튼을 눌러 창을 닫았는지 확인
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("\n창이 닫혀 프로그램을 종료합니다.")
            break

    cv2.destroyAllWindows()
    if current_idx >= num_images:
        print("\n🎉 모든 이미지의 확인을 완료했습니다!")

if __name__ == "__main__":
    main()
