import cv2

# 비디오 로드
cap = cv2.VideoCapture('source/lane_test.mp4')

roi_selected = False
x, y, w, h = 0, 0, 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    # ROI 선택된 경우 사각형 표시 + ROI 출력
    if roi_selected and w > 0 and h > 0:
        # 사각형 한번만 그리기
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_frame = frame[y:y+h, x:x+w]
        if roi_frame.size > 0:
            cv2.imshow("ROI", roi_frame)

    # 전체 영상 출력
    cv2.imshow("Video", display_frame)

    key = cv2.waitKey(1) & 0xFF

    # r 누르면 ROI 선택
    if key == ord('r'):
        roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
        x, y, w, h = roi
        cv2.destroyWindow("Select ROI")

        if w > 0 and h > 0:
            roi_selected = True
            print(f"✅ ROI 좌표: x={x}, y={y}, w={w}, h={h}")
        else:
            print("⚠️ ROI 선택 취소 또는 잘못된 선택")
            roi_selected = False

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
