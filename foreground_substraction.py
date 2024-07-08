import cv2
import numpy as np

cap = cv2.VideoCapture('vtest2.avi')
ret, frame = cap.read()

average = np.float32(frame) # Create a flaot numpy array with frame values

while(True):
    # Get The Frame
    ret, frame = cap.read()

    cv2.accumulateWeighted(frame, average, 0.01) # 0.01 is the weight of image, play around to see how it changes
    fsubtract = cv2.convertScaleAbs(average)

    # Show Video
    cv2.imshow('Frame',frame)
    cv2.imshow('Frame Foreground Substraction',fsubtract)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
