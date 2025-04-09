import cv2
import numpy as np

def empty(_):
    pass

image = cv2.imread("S9_Lambo.png")

cv2.namedWindow("TrackBar")
cv2.resizeWindow("TrackBar", 640, 250)

cv2.createTrackbar("Hue Min", "TrackBar", 0, 179, empty)
cv2.createTrackbar("Saturation Min", "TrackBar", 0, 255, empty)
cv2.createTrackbar("Value Min", "TrackBar", 0, 255, empty)

cv2.createTrackbar("Hue Max", "TrackBar", 179, 179, empty)
cv2.createTrackbar("Saturation Max", "TrackBar", 255, 255, empty)
cv2.createTrackbar("Value Max", "TrackBar", 255, 255, empty)

while True:
    image = cv2.imread("S9_Lambo.png")
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBar")
    s_min = cv2.getTrackbarPos("Saturation Min", "TrackBar")
    v_min = cv2.getTrackbarPos("Value Min", "TrackBar")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBar")
    s_max = cv2.getTrackbarPos("Saturation Max", "TrackBar")
    v_max = cv2.getTrackbarPos("Value Max", "TrackBar")
    print(f"HSV vals: ({h_min}, {s_min}, {v_min}, {h_max}, {s_max}, {v_max})")
    low = np.array([h_min, s_min, v_min])
    up = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv_image, low, up)
    image_result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("image_result", image_result)
    
    # cv2.imshow("frame", img)
    cv2.imshow("TrackBar", image)
    cv2.imshow("image_frame", image)
    cv2.imshow("mask", mask)
    cv2.imshow('HSV', hsv_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows() 