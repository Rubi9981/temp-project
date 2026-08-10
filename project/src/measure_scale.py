import cv2
import numpy as np
import math
import os

# -----------------------------
# 1) 설정
# -----------------------------
W, H = 640, 480
IMG_PATH = 'captures/frame_20260803_162053.jpg'

# 튜닝된 BEV 기준 좌표 [좌상, 우상, 우하, 좌하]
SRC_POINTS = np.array([
    [36, 262],   # 좌상단
    [588, 269],  # 우상단
    [605, 406],  # 우하단
    [17, 413]    # 좌하단
], dtype=np.float32)

# 두 점 사이의 실제 물리적 거리 (cm) - 예: 차선 폭이 40cm 라면 40.0 입력
REAL_DISTANCE_CM = 40.0   # 원하는 실제 센티미터 값으로 변경 가능

# -----------------------------
# 2) BEV 변환
# -----------------------------
def warp_image(image, src_pts, width=W, height=H):
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_points)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped

img = cv2.imread(IMG_PATH)
if img is None:
    # 폴더 내 대체 파일 탐색
    captures_dir = 'captures'
    if os.path.exists(captures_dir):
        files = [f for f in os.listdir(captures_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if files:
            img = cv2.imread(os.path.join(captures_dir, files[0]))

if img is None:
    raise FileNotFoundError("이미지를 열 수 없습니다.")

img = cv2.resize(img, (W, H))
bev_img = warp_image(img, SRC_POINTS, W, H)

# 마우스 클릭 점 저장
pts = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts) >= 2:
            pts.clear()  # 2개 이상이면 초기화 후 첫 점 등록
        pts.append((x, y))
        print(f"점 {len(pts)} 선택: ({x}, {y})")

cv2.namedWindow('Measure 1px to cm (BEV View)')
cv2.setMouseCallback('Measure 1px to cm (BEV View)', on_mouse)

print("=" * 60)
print("마우스로 BEV 화면에서 두 지점(예: 왼쪽 차선과 오른쪽 차선)을 클릭하세요.")
print(f"현재 설정된 실제 거리 기준: {REAL_DISTANCE_CM} cm")
print("  - r: 다시 선택 (초기화)")
print("  - q 또는 ESC: 종료")
print("=" * 60)

while True:
    view = bev_img.copy()

    # 클릭 점 표시
    for i, p in enumerate(pts):
        cv2.circle(view, p, 5, (0, 0, 255), -1)
        cv2.putText(view, f"P{i+1}", (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 2개 점이 찍혔을 때 거리 및 cm/pixel 스케일 계산
    if len(pts) == 2:
        p1, p2 = pts[0], pts[1]
        cv2.line(view, p1, p2, (255, 255, 0), 2)

        # 피타고라스 정리로 픽셀 거리 계산
        pixel_dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        if pixel_dist > 0:
            cm_per_pixel = REAL_DISTANCE_CM / pixel_dist
            pixel_per_cm = pixel_dist / REAL_DISTANCE_CM

            # 화면 상단 정보 상자 표시
            cv2.rectangle(view, (10, 10), (450, 95), (0, 0, 0), -1)
            cv2.putText(view, f"Pixel Distance : {pixel_dist:.2f} px", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(view, f"1 Pixel = {cm_per_pixel:.4f} cm", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(view, f"1 cm = {pixel_per_cm:.2f} Pixels", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow('Measure 1px to cm (BEV View)', view)

    key = cv2.waitKey(20) & 0xFF
    if key == ord('r'):
        pts.clear()
        print("초기화되었습니다.")
    elif key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
