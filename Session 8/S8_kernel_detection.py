import cv2
import numpy as np

image = cv2.imread("Lucy.jpg")
bright_image = cv2.add(image, 150)
contrast_image = cv2.multiply(image, 0.5)

noise = np.random.normal(0, 25, image.shape).astype('float32')
noisy_image = cv2.add(image.astype('float32'), noise)
noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')

kernel = np.ones((3,3), dtype=np.float32) / 9
denoised_image = cv2.filter2D(noisy_image, -1, kernel)

cv2.imshow("denoised", denoised_image)
cv2.imshow("noisy", noisy_image)
# cv2.imshow("noisy", bright_image)
# cv2.imshow("noisy", contrast_image)
cv2.waitKey(0)
cv2.destroyAllWindows()