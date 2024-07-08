import cv2
import glob
 
for filename in glob.glob('./vtest6/*.jpg'):
    frame = cv2.imread(filename)
    # frame_gray = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # Show Video
    cv2.imshow('Frame Play',frame)

    if cv2.waitKey(30) & 0xFF == ord('q'): #quit press q
        break
    
cv2.destroyAllWindows()