import cv2
import numpy as np
import time

webcam = cv2.VideoCapture(0)

time.sleep(1)
frame_count = 0
background = 0

for i in range(60):
    ret, background = webcam.read()
background = np.flip(background, axis=1)

while webcam.isOpened():
    ret, img = webcam.read()

    if not ret:
        break

    frame_count += 1

    img = np.flip(img, axis=1)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([100, 40, 40])
    upper_red = np.array([100, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([155, 40, 40])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask1 = mask1 + mask2

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    mask2 = cv2.bitwise_not(mask1)

    res2 = cv2.bitwise_and(img, img, mask=mask2)

    res1 = cv2.bitwise_and(background, background, mask=mask1)

    finalOutput = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow("magic", finalOutput)
    cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()
