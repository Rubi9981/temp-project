"""차선 이진화 백엔드.

모두 같은 시그니처를 따른다: fn(bgr) -> uint8 mask (0 또는 255).
입력은 BEV 전체 또는 ROI 어느 쪽이든 된다.

hsv_inrange 가 baseline 이다. 개선 주장은 항상 이것과의 비교로만 한다.
"""
import cv2
import numpy as np

import config as cfg


def _morph(mask):
    """OPEN(노이즈 제거) -> CLOSE(끊긴 차선 잇기). 세 백엔드가 공유한다."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, cfg.MORPH_KSIZE)
    if cfg.MORPH_ITER_OPEN > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                iterations=cfg.MORPH_ITER_OPEN)
    if cfg.MORPH_ITER_CLOSE > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                iterations=cfg.MORPH_ITER_CLOSE)
    return mask


def hsv_inrange(bgr):
    """baseline — project/lane_detector_v2.py 방식.

    전역 HSV 임계값이라 조명 그라디언트와 정반사에 취약하다.
    """
    blurred = cv2.GaussianBlur(bgr, cfg.BLUR_KSIZE, 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, cfg.LOWER_HSV, cfg.UPPER_HSV)
    return _morph(mask)


def adaptive_gray(bgr):
    """project/lane_detector.py (v1) 방식 — 국소 적응 임계값.

    조명 그라디언트에는 강하지만, 어두운 노면의 미세한 얼룩까지
    전경으로 올려 과검출이 나기 쉽다.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, cfg.BLUR_KSIZE, 0)

    block = cfg.ADAPT_BLOCK
    if block % 2 == 0:
        block += 1
    block = max(block, 3)

    mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        cfg.ADAPT_C,
    )
    return _morph(mask)


def tophat_otsu(bgr):
    """신규 — LAB의 L채널에 top-hat을 걸어 "얇고 밝은 띠"만 남긴다.

    차선은 커널보다 좁은 밝은 띠, 천장 조명 정반사는 커널보다 넓게 퍼진
    밝은 영역이다. top-hat은 커널보다 큰 밝은 구조를 제거하므로 반사 얼룩이
    걸러진다. 색이 아니라 형태로 구분하는 것이 요점이다.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness = cv2.GaussianBlur(lab[:, :, 0], cfg.BLUR_KSIZE, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, cfg.TOPHAT_KSIZE)
    hat = cv2.morphologyEx(lightness, cv2.MORPH_TOPHAT, kernel)

    thr, mask = cv2.threshold(hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu는 항상 임계값을 하나 고른다. 차선이 아예 없는 프레임에서는
    # 노이즈를 반으로 갈라 전경을 만들어내므로 하한을 둔다.
    if thr < cfg.TOPHAT_MIN_OTSU:
        return np.zeros(mask.shape, dtype=np.uint8)

    return _morph(mask)


BACKENDS = {
    'hsv': hsv_inrange,
    'adaptive': adaptive_gray,
    'tophat': tophat_otsu,
}

BASELINE = 'hsv'
