import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

import warnings
warnings.filterwarnings('ignore')

from mtcnn import MTCNN

IMAGE_SIZE = (32, 32)

detector = MTCNN()

def face_detector(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = detector.detect_faces(img)[0]
    x, y, w, h = out['box']
    
    face = img[y:y+h, x:x+w]
    
    return face

# DATA
def data_preprocess():
    data = []
    labels = []
    
    for i, address in enumerate(glob.glob('smile_dataset/*/*')):
        img = cv2.imread(address)
        face = face_detector(img)
        
        img = cv2.resize(img, IMAGE_SIZE)
        img = img/255
        img = img.flatten()
        
        data.append(img)
        label = address.split('/')[0].split('\\')[1]
        labels.append(label)
        
        if i % 100 == 0:
            print(f'Processed {i} images')
        
    data = np.array(data)
    
    x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, test_size=0.2)
    
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = data_preprocess()

# MODEL
clf = SGDClassifier()
clf.fit(x_train, y_train)


# EVALUATE
predictions = clf.predict(x_test)
print(accuracy_score(x_test))