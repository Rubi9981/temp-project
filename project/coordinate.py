import cv2
import numpy as np

W, H = 640, 480
IMG_PATH = 'captures/frame_20260803_162053.jpg'   # 여기에 이미지 경로 입력

pts = []         # 전체 이미지(640x480) 기준 저장용 좌표
disp_pts = []    # 화면 표시용 (자른 이미지 기준) 좌표

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        # 하단 절반을 잘라냈으므로 원본 Y 좌표는 y + (H // 2)
        real_y = y + (H // 2)
        
        disp_pts.append([x, y])
        pts.append([x, real_y])
        print(f"{len(pts)}번째 점 (화면 클릭: [{x}, {y}] -> 전체 기준: [{x}, {real_y}])")

img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {IMG_PATH}")

img = cv2.resize(img, (W, H))   # 영상 처리 때와 같은 크기로 맞춤
img = img[H // 2 :, :]          # 하단 절반 잘라내기 (y: 240 ~ 480 범위만 표시)

cv2.namedWindow('Pick')
cv2.setMouseCallback('Pick', on_mouse)

print("좌상 → 우상 → 우하 → 좌하 순서로 4번 클릭 (r: 초기화, q: 종료)")

while True:
    view = img.copy()

    for i, p in enumerate(disp_pts):
        cv2.circle(view, tuple(p), 5, (0, 0, 255), -1)
        cv2.putText(view, str(i + 1), (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if len(disp_pts) == 4:
        cv2.polylines(view, [np.array(disp_pts)], True, (0, 255, 0), 2)

    cv2.imshow('Pick', view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):        # 다시 찍기
        pts.clear()
        disp_pts.clear()
        print("초기화")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()

if len(pts) == 4:
    print("\nsrc = np.float32([" + ", ".join(str(p) for p in pts) + "])")