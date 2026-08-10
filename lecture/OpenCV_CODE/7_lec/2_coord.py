import cv2

# 마우스 이벤트 콜백 함수
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked coordinates: ({x}, {y})")

# 영상 파일 로드
cap = cv2.VideoCapture('source/lane_test.mp4') 
fps = cap.get(cv2.CAP_PROP_FPS)
# 창에 마우스 콜백 함수 설정
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 크기를 640x480으로 고정 리사이즈
    frame = cv2.resize(frame, (640, 480))

    # 결과 출력
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
