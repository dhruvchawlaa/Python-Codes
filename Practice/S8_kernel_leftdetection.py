import cv2
import numpy as np

image = cv2.imread("Lucy.jpg")
noise = np.random.normal(0, 25, image.shape).astype('float32')
noisy_image = cv2.add(image.astype('float32'), noise)
noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')

cv2.imshow("frame", noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()