import cv2
import numpy as np
from tensorflow.keras.models import load_model

net = load_model("../Session 23/kapcha_model_decoder.h5")
img = cv2.imread("digits.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)
cntr1, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, cntr1, -1, (255, 0, 0), 2)

for i in range(len(cntr1)):
    x, y, w, h = cv2.boundingRect(cntr1[i])
    
    roi = img[y-5 : y+h+5, x-5 : x+w+5]
    roi = cv2.resize(roi, (32, 32))
    roi = roi/255
    roi = np.array([roi])
    prediction = net.predict(roi)
    # print(prediction)
    pred = np.argmax(prediction[0])
    print(pred+1)
    
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(img, str(pred+1), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("frame", img)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()
    
