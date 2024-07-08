import cv2
import numpy as np

cap = cv2.VideoCapture('vtest5.mp4')

# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbg2 = cv2.bgsegm.createBackgroundSubtractorMOG()

fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg2 = cv2.createBackgroundSubtractorMOG2()

a = np.uint8([255])
b = np.uint8([0])

while(True):
    # Get The Frame
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.imshow('Frame',frame)

    # Apply Method on Frame with different learning rate

    fgmask_sort = fgbg.apply(frame_gray, learningRate = 0.005)
    cv2.imshow('Frame Short Term',fgmask_sort)

    fgmask_long = fgbg2.apply(frame_gray, learningRate = 0.00005)
    cv2.imshow('Frame Long Term',fgmask_long)

    # foreground defferences
    res = cv2.absdiff(fgmask_long,fgmask_sort)
    # res = fgmask_long - fgmask_sort
    res = np.where((255==res),a,b) ##-------> harus ada kondisi dimana menampilkan pixel objek saja tidak bg
    cv2.imshow('foreground defferences',res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()