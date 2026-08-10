import cv2
import numpy as np
import os

# -----------------------------
# 1) 설정 (이미지 경로 및 창 크기)
# -----------------------------
W, H = 640, 480
IMG_PATH = 'captures_bev/bev_frame_20260803_162132.jpg'  # 분석할 이미지 경로

# 클릭 정보 저장용 변수
clicked_pos = None
hsv_val = None
bgr_val = None

def on_mouse(event, x, y, flags, param):
    global clicked_pos, hsv_val, bgr_val
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pos = (x, y)
        bgr_val = img[y, x]
        hsv_val = hsv_img[y, x]
        
        h, s, v = hsv_val
        b, g, r = bgr_val
        print(f"[클릭 위치 ({x}, {y})] -> HSV: ({h}, {s}, {v}) | BGR: ({b}, {g}, {r})")

# -----------------------------
# 2) 이미지 불러오기 및 HSV 변환
# -----------------------------
if not os.path.exists(IMG_PATH):
    # 폴더 내 다른 이미지가 있는지 확인
    captures_dir = 'captures'
    if os.path.exists(captures_dir):
        files = [f for f in os.listdir(captures_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if files:
            IMG_PATH = os.path.join(captures_dir, files[0])

img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {IMG_PATH}")

# 분석하기 편하게 리사이즈
img = cv2.resize(img, (W, H))

# BGR -> HSV 변환
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow('HSV Picker')
cv2.setMouseCallback('HSV Picker', on_mouse)

print("=" * 60)
print(f"이미지: {IMG_PATH}")
print("마우스 왼쪽 버튼으로 궁금한 지점을 클릭하면 HSV 및 BGR 값을 확인할 수 있습니다.")
print("  - r: 클릭 위치 초기화")
print("  - q 또는 ESC: 종료")
print("=" * 60)

# -----------------------------
# 3) 실시간 시각화 루프
# -----------------------------
while True:
    view = img.copy()

    if clicked_pos is not None and hsv_val is not None:
        x, y = clicked_pos
        h, s, v = hsv_val
        b, g, r = bgr_val

        # 클릭한 위치에 조그만 원 표시
        cv2.circle(view, (x, y), 5, (0, 0, 255), -1)

        # HSV 및 BGR 정보 가독성 높게 텍스트 표시
        text_hsv = f"HSV: H={h}, S={s}, V={v}"
        text_bgr = f"BGR: B={b}, G={g}, R={r}"
        
        # 텍스트 배경 박스 (상단)
        cv2.rectangle(view, (10, 10), (320, 75), (0, 0, 0), -1)
        cv2.putText(view, text_hsv, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(view, text_bgr, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # 클릭 위치 옆에도 조그맣게 표시
        text_pos = f"({x},{y})"
        cv2.putText(view, text_pos, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('HSV Picker', view)

    key = cv2.waitKey(20) & 0xFF
    if key == ord('r'):
        clicked_pos = None
        hsv_val = None
        bgr_val = None
        print("초기화되었습니다.")
    elif key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
