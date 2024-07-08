import numpy as np
import cv2

# function for trackbar
def nothing(x):
	pass

cap = cv2.VideoCapture('vtest.avi')
images = [] #array to store frame

cv2.namedWindow('BS Mean Filtering')
cv2.createTrackbar('treshold','BS Mean Filtering',50,255,nothing) #trackbar start in 50
while True:
	
	ret,frame = cap.read()
	cv2.imshow('image',frame)

    #change size frame
	dim = (500,500)
	frame = cv2.resize(frame,dim,interpolation = cv2.INTER_AREA) 
    
    #converting images into grayscale       
	frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #append frame_gray into images array
	images.append(frame_gray)

	# removing the images after every 50 image
	if len(images)==50:
	        images.pop(0)

    #buat array baru 'image' sama dengan 'images' 
	image = np.array(images)

	# gettting the tracker value
	treshold = cv2.getTrackbarPos('treshold','BS Mean Filtering')

    #Mean n previous image
	image = np.mean(image,axis=0)

	#ubah tipe ke uint8
	image = image.astype(np.uint8)

    #show background
	cv2.imshow('background',image)

	# foreground will be background - curr frame
	foreground_image = cv2.absdiff(frame_gray,image) # current image - mean(N previous image)

	a = np.array([0],np.uint8)
	b = np.array([255],np.uint8)

	img = np.where(foreground_image>treshold,b,a) #untuk hanya objek putih 'frame_gray' ganti b
	cv2.imshow('BS Mean Filtering',img)

	if cv2.waitKey(1) & 0xFF == 27:
		break


cap.release()

cv2.destroyAllWindows()		