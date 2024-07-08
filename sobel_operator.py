import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('sobel_try.png')

# Ubah ukuran gambar
#///////////////////////////////////////////////////////////////
scale_percent = 100 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#///////////////////////////////////////////////////////////////

# change image to grayscale
gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Sobel Operator

# img_sobelx = cv2.Sobel(image,cv2.CV_8U,1,0,ksize=5)
# img_sobely = cv2.Sobel(image,cv2.CV_8U,0,1,ksize=5)
# img_sobel = img_sobelx + img_sobely

# define filters
horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # s2
vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # s1

# define images with 0s
img_sobelx = np.zeros((height, width))
img_sobely = np.zeros((height, width))
img_sobelxy = np.zeros((height, width))

for i in range(1, height - 1):
    for j in range(1, width - 1):
        horizontalGrad = (horizontal[0, 0] * gray_img[i - 1, j - 1]) + \
                         (horizontal[0, 1] * gray_img[i - 1, j]) + \
                         (horizontal[0, 2] * gray_img[i - 1, j + 1]) + \
                         (horizontal[1, 0] * gray_img[i, j - 1]) + \
                         (horizontal[1, 1] * gray_img[i, j]) + \
                         (horizontal[1, 2] * gray_img[i, j + 1]) + \
                         (horizontal[2, 0] * gray_img[i + 1, j - 1]) + \
                         (horizontal[2, 1] * gray_img[i + 1, j]) + \
                         (horizontal[2, 2] * gray_img[i + 1, j + 1])

        img_sobelx[i - 1, j - 1] = abs(horizontalGrad)

        verticalGrad = (vertical[0, 0] * gray_img[i - 1, j - 1]) + \
                       (vertical[0, 1] * gray_img[i - 1, j]) + \
                       (vertical[0, 2] * gray_img[i - 1, j + 1]) + \
                       (vertical[1, 0] * gray_img[i, j - 1]) + \
                       (vertical[1, 1] * gray_img[i, j]) + \
                       (vertical[1, 2] * gray_img[i, j + 1]) + \
                       (vertical[2, 0] * gray_img[i + 1, j - 1]) + \
                       (vertical[2, 1] * gray_img[i + 1, j]) + \
                       (vertical[2, 2] * gray_img[i + 1, j + 1])

        img_sobely[i - 1, j - 1] = abs(verticalGrad)

        # Edge Magnitude
        mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
        img_sobelxy[i - 1, j - 1] = mag
        teta = np.arctan(img_sobely/img_sobelx) #untuk menghitung teta = arctan(Gy/Gx)

# show image
cv2.imshow('image',image)
cv2.imshow('image grayscale', gray_img)
cv2.imshow('sobel x',img_sobelx)
cv2.imshow('sobel y',img_sobely)
cv2.imshow('sobel x+y',img_sobelxy)

# print(teta)

cv2.waitKey(0)
cv2.destroyAllWindows()