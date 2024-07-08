import cv2

cap = cv2.VideoCapture('vtest.avi')

while(True):
    # Get The Frame
    ret, frame = cap.read()

    # Show Video
    cv2.imshow('Frame',frame)

    if cv2.waitKey(30) & 0xFF == ord('q'): #quit press q
        break

cap.release()
cv2.destroyAllWindows()
