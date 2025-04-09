import cv2
import numpy as np
from keras import models

IMG_SIZE = (32, 32)
labels = ['fire', 'nonfire']

img = cv2.imread('test3.jpg')
img = cv2.resize(img, IMG_SIZE)
img = img / 255.0
img = np.expand_dims(img, axis=0) 

net = models.load_model('classification.h5')
pred = net.predict(img)
print("Prediction values:", pred)

max_pred = np.argmax(pred)
out = labels[max_pred]

print("Prediction:", out)
