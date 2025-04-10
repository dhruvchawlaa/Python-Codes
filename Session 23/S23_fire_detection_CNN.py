import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

import warnings
warnings.filterwarnings('ignore')
from joblib import dump

def load_data():
    data_list = []
    labels = []
    le = LabelEncoder()

    for i, address in enumerate(glob.glob('../Practice/fire_dataset//*//*')):
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

aug = ImageDataGenerator(rotation_range=20,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")

# MODEL
net = models.Sequential([
                         layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                         layers.BatchNormalization(),
                         layers.MaxPooling2D((2, 2)),
                         layers.Conv2D(64, (3, 3), activation='relu'),
                        #  layers.BatchNormalization(),
                         layers.MaxPooling2D((2, 2)),
                         layers.Flatten(),
                        #  layers.BatchNormalization(),
                         layers.Dense(2, activation='softmax'),
                        ])

# SGD(learning_rate=0.001)

net.compile(optimizer='SGD',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

H = net.fit(aug.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), batch_size=64, epochs=30)


plt.plot(H.history["accuracy"], label="train")
plt.plot(H.history["val_accuracy"], label="test")
plt.plot(H.history["loss"], label="train loss")
plt.plot(H.history["val_loss"], label="test loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.show()

net.save('classification.h5')