import cv2
import numpy as np

image= cv2.imread("square.png")

kernel = np.array([[-1, 1]])

filter_  = cv2.filter2D(image, cv2.CV_8U, kernel)

cv2.imshow("frame", filter_)
cv2.waitKey(0)
cv2.destroyAllWindows()