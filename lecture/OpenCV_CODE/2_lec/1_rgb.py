import cv2
import numpy as np
import matplotlib.pyplot as plt

image = [[[[0, 0, 255], [0, 255, 0]],
          [[255, 0, 0], [0, 255, 255]]]]
image = np.array(image, dtype=np.uint8) 
image = image[0]  

image = cv2.resize(image, (480, 480),interpolation=cv2.INTER_NEAREST)
#image = cv2.resize(image, (480, 480),interpolation=cv2.INTER_LINEAR)
#image = cv2.resize(image, (480, 480),interpolation=cv2.INTER_CUBIC)

img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cv2.imshow("image", img_rgb)                   
cv2.waitKey(0)                                
cv2.destroyAllWindows()      

# matplotlib로 출력
plt.figure(figsize=(3, 3))
plt.imshow(img_rgb)
plt.title("RGB Image (matplotlib)")
plt.axis("off")
plt.show()