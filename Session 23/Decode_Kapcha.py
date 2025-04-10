
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential, save_model

def data_preprocessing():
    images =[]
    labels = []
    
    for i, item in enumerate(glob.glob("..\Practice\kapcha\\*\\*")):
        img = cv2.imread(item)
        img = cv2.resize(img, (32, 32))
        img = img / 255
        images.append(img)
        
        label = item.split("\\")[3]
        labels.append(label)
        
        if i % 100 == 0:
            print(f"Processed {i} images")
        
    images = np.array(images)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, shuffle=True)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    return X_train, X_test, y_train, y_test

def classification_model(X_train, X_test, y_train, y_test):
    net = Sequential([
                      Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                      MaxPooling2D((2, 2)),
                      Conv2D(32, (3, 3), activation='relu'),
                      MaxPooling2D((2, 2)),
                      Flatten(),
                      Dense(16, activation='relu'),
                      Dense(9, activation='softmax'),
    ])
    
    net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    H = net.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
    save_model(net, "kapcha_model_decoder.h5")
    return H

def visualize_results(H):
    plt.plot(H.history['loss'], label='train loss')
    plt.plot(H.history['val_loss'], label='validation loss')
    plt.plot(H.history['accuracy'], label='train acc')
    plt.plot(H.history['val_accuracy'], label='test acc')
    plt.xlabel('epoch')
    plt.ylabel('metrics')
    plt.title('EVALUATION')
    plt.legend()
    plt.show()
    
        
X_train, X_test, y_train, y_test = data_preprocessing()
H = classification_model(X_train, X_test, y_train, y_test)
visualize_results(H)