import cv2
import time
from ultralytics import YOLO

# YOLOv8 모델 로드 (pt 파일 경로를 설정)
model = YOLO('3_lec/best (1).pt')  # 여기서 pt 파일 경로를 지정

# 웹캠에서 실시간 스트리밍을 위한 비디오 캡처 설정
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미

while True:
    start_time = time.time()  # 프레임 시작 시간 기록
    
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv8 모델로 프레임 예측
    results = model(frame)
    
    # 결과를 이미지에 그리기
    annotated_frame = results[0].plot()  # 예측된 객체를 프레임에 표시
    
    # FPS 계산
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    
    # FPS 텍스트 표시
    cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 결과를 화면에 표시
    cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)
    
    # 'q'를 눌러 스트리밍을 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 해제 및 모든 창 닫기
cap.release()
cv2.destroyAllWindows()
