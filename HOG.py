import cv2
from skimage.feature import hog
from skimage import exposure

img = cv2.imread('car.jpg')
cv2.imshow('image', img)

h, w, c = img.shape

MC = True # for color image
# MC = False # for grayscale image

# ekstrasi fitur hog. hog_desc = fitur hog , hog_image = visualisasi dari fitur hog
# orientations=9 adalah bin histogramnya (0, 20, 40, 60, 80, 100, 120, 140, 160, 180) derajat
# block_norm='L2-Hys' (normalisasi) L2-Hys (default)
# Block normalization method:
# L1 = Normalization using L1-norm.
# L1-sqrt = Normalization using L1-norm, followed by square root.
# L2 = Normalization using L2-norm.
# L2-Hys = Normalization using L2-norm, followed by limiting the maximum values to 0.2 (Hys stands for hysteresis) and renormalization using L2-norm.
# hog_desc, hog_image = hog(img, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), visualize=True, multichannel=MC)

hog_desc, hog_image = hog(img, orientations=9, pixels_per_cell=(h/20,w/20), cells_per_block=(2,2), visualize=True, multichannel=MC)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0,5)) #untuk rescale intensity value to range
cv2.imshow('image HOG', hog_image_rescaled)

cv2.waitKey(0)
cv2.destroyAllWindows()