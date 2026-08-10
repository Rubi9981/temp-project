import cv2

# 이미지 읽기
img = cv2.imread('source/lane.jpg')

# ROI 선택
x, y, w, h = cv2.selectROI('img', img, False)
if w and h:
    roi = img[y:y+h, x:x+w]
    
    # ROI 정보 출력
    print(f"ROI 좌표: x={x}, y={y}, w={w}, h={h}")
    
    # ROI 지정 영역을 새창으로 표시
    cv2.imshow('cropped', roi)
    
    # 새창을 화면 좌측 상단에 이동
    cv2.moveWindow('cropped', 0, 0)
    
    # ROI 영역만 파일로 저장
    cv2.imwrite('source/cropped.jpg', roi)

# 키 입력 대기 및 윈도우 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()
