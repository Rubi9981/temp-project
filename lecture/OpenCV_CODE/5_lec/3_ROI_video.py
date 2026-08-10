import cv2
import numpy as np

# 1. 웹캠 열기 및 해상도 설정
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 2. 고정된 사각형 ROI 좌표 정의
# 예시: 영상 하단 중앙 사각형 (가로 200, 세로 100)
roi_points = np.array([
    [220, 380],   # 좌하단
    [420, 380],   # 우하단
    [420, 280],   # 우상단
    [220, 280]    # 좌상단
], dtype=np.int32)

while True:
    ret, frame = cap.read()

    # 3. 빈 마스크 생성 (frame과 동일한 크기, 흑백)
    roi_mask = np.zeros((480, 640), dtype=np.uint8)

    # 4. ROI 영역만 흰색(255)으로 채움
    cv2.fillPoly(roi_mask, [roi_points], 255)

    # 5. 마스크 적용하여 ROI 부분만 추출
    roi_result = cv2.bitwise_and(frame, frame, mask=roi_mask)

    # 6. ROI 윤곽선을 원본 프레임 위에 그리기
    cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)

    # 7. 결과 시각화
    cv2.imshow("Original Frame with ROI Outline", frame)
    cv2.imshow("ROI Mask", roi_mask)
    cv2.imshow("ROI Applied Result", roi_result)

    # 종료 조건: 'q' 키
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. 정리
cap.release()
cv2.destroyAllWindows()
