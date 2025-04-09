import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from joblib import dump

def load_data():
    data_list = []
    labels = []

    for i, address in enumerate(glob.glob('fire_dataset\\*\\*')):
        img = cv2.imread(address)
        if img is None:
            print(f"[WARNING]: Could not read image at {address}")
            continue  # Skip this file
    
        img = cv2.resize(img, (32, 32))
        img = img / 255
        img = img.flatten()

        data_list.append(img)
        label = address.split("\\")[-1].split(".")[0]
        labels.append(label)

        if i % 100 == 0:
            print(f"[INFO]: {i}/1000 processed")
    
    data_list = np.array(data_list)
    
    X_train, X_test, y_train, y_test = train_test_split(data_list, labels, test_size=0.2)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

clf_LR = LogisticRegression()
clf_LR.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))

predictions_LR = clf_LR.predict(X_test)
print(accuracy_score(y_test, predictions_LR))

dump(clf, 'fire_detection.z')