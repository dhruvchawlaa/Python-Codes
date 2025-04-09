import cv2
import numpy as np

image = cv2.imread("Lucy.jpg")
'''
if image is None:
    print("Could not open or find the image")
    
# Gaussian Noise
# image + noise
gaussian_noise = np.random.normal(0, 25, image.shape).astype('float32')
noisy_image = cv2.add(image.astype('float32'), gaussian_noise)
noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')
    
cv2.imshow('frame', noisy_image)
cv2.waitKey(0)
'''

def add_noise(img: np.asarray) -> np.asarray:
    row, col, _ = img.shape
    number_of_pixel = np.random.randint(300, 20000)
    for i in range(number_of_pixel):
        x = np.random.randint(0, row-1)
        y = np.random.randint(0, col-1)
        img[x, y] = 0
        
    for i in range(number_of_pixel):
        x = np.random.randint(0, row-1)
        y = np.random.randint(0, col-1)
        img[x, y] = 255
        
    return img

noisy_image = add_noise(image)
cv2.imshow('frame', noisy_image)
cv2.waitKey(0)