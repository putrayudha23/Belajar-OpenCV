import cv2
import numpy as np

cap = cv2.VideoCapture('vtest5.mp4')

# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbg2 = cv2.bgsegm.createBackgroundSubtractorMOG()

fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=5, detectShadows=True)
fgbg2 = cv2.createBackgroundSubtractorMOG2(varThreshold=5, detectShadows=True)

a = np.uint8([255])
b = np.uint8([0])

#get fps
f = cap.get(cv2.CAP_PROP_FPS)

#parameter
s = 10.0
th = 0.75
k = f*s

while(True):
    # Get The Frame
    ret, frame = cap.read()
    cv2.imshow('Frame', frame)
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    ################################################################

    #https://stackoverflow.com/questions/46000390/opencv-backgroundsubtractor-yields-poor-results-on-objects-with-similar-color-as
    frame_gray = cv2.equalizeHist(frame_gray) #Histogram Equalization to overcome same colour distribution

    # CHANGE TO HSV -> GET HEU CHANNEL

    _, mask = cv2.threshold(frame_gray, 220, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    #Morphological Transformations https://www.youtube.com/watch?v=xSzsD4kXhRw
    # frame_gray = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # frame_gray = cv2.erode(mask, kernel, iterations=1)
    # frame_gray = cv2.dilate(mask, kernel, iterations=1)

    ################################################################

    # Apply Method on Frame with different learning rate

    fgmask_sort = fgbg.apply(frame_gray, learningRate = 0.005) #mog2 0.005
    cv2.imshow('Frame Short Term',fgmask_sort)

    fgmask_long = fgbg2.apply(frame_gray, learningRate = 0.00005) #mog2 0.00005
    cv2.imshow('Frame Long Term',fgmask_long)

    # foreground differences
    ''' cv2.absdiff is a function which helps in finding the absolute difference between
    the pixels of the two image arrays. By using this we will be able to extract
    just the pixels of the objects that are moving. '''

    res = cv2.absdiff(fgmask_long,fgmask_sort)

    # cumulative value of foreground difference
    
    # 1==res artinya hanya menampilkan pixel 1
    res = np.where((255==res),a,b) ##-------> harus ada kondisi dimana menampilkan pixel objek saja tidak bg
    cv2.imshow('foreground defferences',res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()