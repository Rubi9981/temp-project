import cv2
import numpy as np
import os

# -----------------------------
# 1) 설정 및 파라미터
# -----------------------------
W, H = 640, 480

# coordinate.py에서 구한 최신 BEV 기준 좌표 [좌상, 우상, 우하, 좌하]
SRC_POINTS = np.array([
    [36, 262],   # 좌상단
    [588, 269],  # 우상단
    [605, 406],  # 우하단
    [17, 413]    # 좌하단
], dtype=np.float32)

INPUT_DIR = 'captures'              # 원본 이미지 폴더
OUTPUT_DIR = 'captures_bev'         # BEV 변환 이미지 저장 폴더
SINGLE_IMG_PATH = 'captures/frame_20260803_162053.jpg'  # 개별 저장용 경로

# -----------------------------
# 2) BEV 원근 변환 함수
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


# -----------------------------
# 3) 이미지 저장 처리 함수
# -----------------------------
def save_single_bev(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 이미지를 찾을 수 없습니다: {img_path}")
        return False

    img = cv2.resize(img, (W, H))
    bev_img = warp_image(img, SRC_POINTS, W, H)

    # 출력 폴더 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # cv2.imwrite로 저장
    success = cv2.imwrite(save_path, bev_img)
    if success:
        print(f"✅ BEV 이미지 저장 완료: {save_path}")
    else:
        print(f"❌ 저장 실패: {save_path}")
    return success


def save_all_captures_bev(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"❌ 원본 폴더가 존재하지 않습니다: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"총 {len(files)}개 이미지의 BEV 변환 저장을 시작합니다...")

    saved_count = 0
    for filename in files:
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, f"bev_{filename}")

        img = cv2.imread(in_path)
        if img is None:
            continue

        img = cv2.resize(img, (W, H))
        bev_img = warp_image(img, SRC_POINTS, W, H)

        if cv2.imwrite(out_path, bev_img):
            saved_count += 1

    print(f"🎉 일괄 저장 완료: {saved_count}/{len(files)} 개 파일이 '{output_dir}/' 폴더에 저장되었습니다.")


# -----------------------------
# 4) 메인 실행
# -----------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("1. 단일 이미지 BEV 변환 저장")
    print("2. captures 폴더 내 전체 이미지 BEV 변환 일괄 저장")
    print("=" * 60)

    # 1) 단일 파일 저장 테스트 (bev_output.jpg로 저장)
    save_single_bev(SINGLE_IMG_PATH, 'captures/bev_output.jpg')

    # 2) captures 폴더 전체 일괄 저장 (captures_bev 폴더에 저장)
    print("\n------------------------------------------------------------")
    save_all_captures_bev(INPUT_DIR, OUTPUT_DIR)
