import cv2
import numpy as np
from tensorflow.keras.models import load_model

### 1단계 : MNIST 모델 로드 ###
model = load_model('L_7/mnist_cnn_model.h5')



### 2단계 : 비디오 캡처 설정 ###
cap = cv2.VideoCapture(0)  # 웹캠 사용
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()



### 3단계 : ROI 영역 좌표 설정 ###
roi_points = np.array([
    [220, 120],    # 좌상단
    [420, 120],    # 우상단
    [420, 320],    # 우하단
    [220, 320]     # 좌하단
], dtype=np.int32)



while True:
    ### 4단계 : 프레임 읽기 ###
    ret, frame = cap.read()
    if not ret:
        break



    ### 5단계 : ROI 마스크 생성 ###
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)



    ### 6단계 : ROI 시각화 ###
    roi_highlight = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.polylines(frame, [roi_points], isClosed=True, color=(255, 255, 0), thickness=2)



    ### 7단계 : ROI 이미지 자르기 ###
    x_start, y_start = roi_points[0]
    x_end, y_end = roi_points[2]
    roi_cropped = frame[y_start:y_end, x_start:x_end]



    ### 8단계 : 흑백 변환 및 28x28 리사이즈 ###
    gray = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    input_img = resized.astype(np.float32) / 255.0
    input_img = np.expand_dims(input_img, axis=(0, -1))  # (1, 28, 28, 1)



    ### 9단계 : 숫자 예측 ###
    prediction = model.predict(input_img, verbose=0)
    pred_digit = np.argmax(prediction)
    confidence = np.max(prediction)



    ### 10단계 : 예측 결과 시각화 ###
    label = f"{pred_digit} ({confidence:.2f})"
    cv2.putText(frame, label, (x_start, y_start - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)



    ### 11단계 : 출력 ###
    cv2.imshow('ROI Only', roi_highlight)
    cv2.imshow('MNIST Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



### 12단계 : 종료 처리 ###
cap.release()
cv2.destroyAllWindows()
