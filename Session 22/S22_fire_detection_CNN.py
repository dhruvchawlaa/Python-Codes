import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras import models, layers

import warnings
warnings.filterwarnings('ignore')
from joblib import dump

def load_data():
    data_list = []
    labels = []
    le = LabelEncoder()

    for i, address in enumerate(glob.glob('fire_dataset\\*\\*')):
        img = cv2.imread(address)
        if img is None:
            print(f"[WARNING]: Could not read image at {address}")
            continue  # Skip this file
    
        img = cv2.resize(img, (32, 32))
        img = img / 255
        # img = img.flatten()

        data_list.append(img)
        label = address.split("\\")[-1].split(".")[0]
        labels.append(label)

        if i % 100 == 0:
            print(f"[INFO]: {i}/1000 processed")
    
    data_list = np.array(data_list)
    
    X_train, X_test, y_train, y_test = train_test_split(data_list, labels, test_size=0.2)
    
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    print(f'before one hot: {y_train}')

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    print(f'after one hot: {y_train}')
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()

# MODEL
net = models.Sequential([
                         layers.Conv2D(32, (3, 3), activation='relu'),
                         layers.MaxPooling2D((2, 2)),
                         layers.Conv2D(64, (3, 3), activation='relu'),
                         layers.MaxPooling2D((2, 2)),
                         layers.Flatten(),
                         layers.Dense(2, activation='softmax'),
                        ])

net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
net.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
net.save('classification.h5')