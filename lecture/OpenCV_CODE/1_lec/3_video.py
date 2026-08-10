import cv2

# 웹캠 열기 (기본 카메라: 0)
#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture('source/lane_test.mp4')

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    # 프레임 화면에 표시
    cv2.imshow('Webcam Live', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
