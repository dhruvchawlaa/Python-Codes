import cv2
import numpy as np
from keras import models

IMG_SIZE = (32, 32)

labels = ['fire', 'nonfire']

img = cv2.imread('test2.jpg')
img = cv2.resize(img, IMG_SIZE)
img = img/255 
img = img.flatten()
img = np.array([img])

net = models.load_model('classification.h5')
pred = net.predict(img)
print(pred)
max_pred = np.argmax(pred)
out = labels[max_pred]

print(out)


