import cv2
import numpy as np

cap = cv2.VideoCapture('vtest5.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

#untuk menampilkan labels hasil proses CCL
def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR) #rgb juga bisa

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('CCL', labeled_img)

while(True):
    # Get The Frame
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(frame_gray, learningRate = 0.00005)
    thresh, biner = cv2.threshold(fgmask, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Show Video
    cv2.imshow('Frame',frame)
    cv2.imshow('Frame Gray',frame_gray)
    cv2.imshow('Frame binary',biner)
    cv2.imshow('Frame BackgroundSubtractorMOG2',fgmask)

    #CCL
    connectivity = 4  # You need to choose 4 or 8 for connectivity type
    num_labels, labels, stats, cetroids = cv2.connectedComponentsWithStats(biner, connectivity, cv2.CV_32S)

    # 1. num_laberls: The total number of unique labels (i.e., number of total components) that were detected
    # 2. A mask named labels has the same spatial dimensions as our input thresh image. For each location in labels,
    # we have an integer ID value that corresponds to the connected component where the pixel belongs. 
    # You’ll learn how to filter the labels matrix later in this section.
    # 3. stats: Statistics on each connected component, including the bounding box coordinates and area (in pixels).
    # 4. The centroids (i.e., center) (x, y)-coordinates of each connected component.

    imshow_components(labels)

    if cv2.waitKey(30) & 0xFF == ord('q'): #quit press q
        break

cap.release()
cv2.destroyAllWindows()