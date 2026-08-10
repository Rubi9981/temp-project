import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드 (pt 파일 경로를 설정)
model = YOLO('3_lec/best.pt')

# 웹캠에서 실시간 스트리밍을 위한 비디오 캡처 설정
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to grab frame")
        break
    
    # YOLO26 모델로 프레임 예측
    results = model(frame, conf=0.3)
    
    # 결과를 이미지에 그리기
    annotated_frame = results[0].plot()  # 예측된 객체를 프레임에 표시

    # 결과를 화면에 표시
    cv2.imshow('YOLO26 Real-time Detection', annotated_frame)
    
    # 'q'를 눌러 스트리밍을 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 해제 및 모든 창 닫기
cap.release()
cv2.destroyAllWindows()
