import cv2
import numpy as np

cap = cv2.VideoCapture('vtest.avi')

#"An improved adaptive background mixture model for real-time tracking with shadow detection” by P. KadewTraKuPong and R. Bowden in 2001
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

#“Improved adaptive Gausian mixture model for background subtraction” in 2004 and “Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction” in 2006
fgbg2 = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

#“Visual Tracking of Human Visitors under Variable-Lighting Conditions for a Responsive Audio Art Installation” in 2012
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG()

#Background substraction KNN
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# fgbg4 = cv2.createBackgroundSubtractorKNN()

while(True):
    # Get The Frame
    ret, frame = cap.read()

    # Apply Method on Frame
    # fgmask = fgbg.apply(frame)
    fgmask2 = fgbg2.apply(frame)

    for y in range(0,int(cap.get(4))):
        for x in range(0,int(cap.get(3))):
            if fgmask2[y,x] == 255 or fgmask2[y,x] == 0:
                fgmask2[y,x] = 0
            else:
                fgmask2[y,x] = 255
    
    # fgmask3 = fgbg3.apply(frame)
    # fgmask3 = cv2.morphologyEx(fgmask3, cv2.MORPH_OPEN, kernel)

    # fgmask4 = fgbg4.apply(frame)
    # fgmask4 = cv2.morphologyEx(fgmask4, cv2.MORPH_OPEN, kernel)

    # Show Video
    cv2.imshow('Frame',frame)
    # cv2.imshow('Frame BackgroundSubtractorMOG',fgmask)
    cv2.imshow('Frame BackgroundSubtractorMOG2',fgmask2)
    # print(min(fgmask2))
    # cv2.imshow('Frame BackgroundSubtractorGMG',fgmask3)
    # cv2.imshow('Frame BackgroundSubtractorKNN',fgmask4)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
