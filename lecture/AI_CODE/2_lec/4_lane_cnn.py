import cv2
import numpy as np
from tensorflow.keras.models import load_model

### 1단계 : 모델 로드 및 클래스 정의 ###
model = load_model('2_lec/cnn_model_gen_noaug.h5')
class_names = ['go', 'left', 'right']



### 2단계 : 비디오 캡처 설정 ###
cap = cv2.VideoCapture(0)  # 웹캠 사용
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()



### 3단계 : ROI 영역 좌표 설정 ###
roi_points = np.array([
    [0, 239],    # 좌상단
    [639, 239],    # 우상단
    [639, 479],   # 우하단
    [0, 479]    # 좌하단
], dtype=np.int32)



while True:
    ### 4단계 : 프레임 읽기 ###
    ret, frame = cap.read()
    if not ret:
        break

    ### 5단계 : ROI 마스크 생성 ###
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)  # ROI 내부를 흰색(255)으로 채움

    ### 6단계 : ROI 영역 강조 ###
    roi_highlight = cv2.bitwise_and(frame, frame, mask=mask)

    ### 7단계 : ROI 사각형 외곽선 그리기 ###
    cv2.polylines(frame, [roi_points], isClosed=True, color=(255, 255, 0), thickness=2)



    ### 8단계 : 모델 입력용 ROI 이미지 잘라내기 ###
    x_start, y_start = roi_points[0]
    x_end, y_end = roi_points[2]
    roi_cropped = frame[y_start:y_end, x_start:x_end]

    input_img = cv2.resize(roi_cropped, (64, 64))           # 모델 입력 사이즈
    input_img = input_img.astype(np.float32) / 255.0        # 정규화
    input_img = np.expand_dims(input_img, axis=0)           # (1, 64, 64, 3)



    ### 9단계 : CNN 예측 ###
    prediction = model.predict(input_img, verbose=0)
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)



    ### 10단계 : 예측 결과 시각화 ###
    label = f"{pred_class.upper()} ({confidence:.2f})"
    cv2.putText(frame, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)



    ### 11단계 : 화면 출력 ###
    cv2.imshow('ROI Only', roi_highlight)
    cv2.imshow('Prediction with ROI Box', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



### 12단계 : 종료 처리 ###
cap.release()
cv2.destroyAllWindows()
