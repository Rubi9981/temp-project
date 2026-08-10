import cv2
import numpy as np

# 이미지 읽기 및 전처리
img = cv2.imread('source/shape_rect.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

# 컨투어 찾기
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 컨투어 그리기 및 특징 계산
for contour in contours:
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    
    # 면적 계산
    area = cv2.contourArea(contour)
    
    # 둘레 계산
    perimeter = cv2.arcLength(contour, True)
    
    # 모멘트 계산
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = 0, 0
    
    # 특징 텍스트로 표시
    text = f"Area: {int(area)}, Perimeter: {int(perimeter)}, Center: ({cx},{cy})"
    cv2.putText(img, text, (cx - 100, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# 결과 이미지 출력
cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
