import cv2 #untuk import lybrary OpenCV

gambar = cv2.imread("Cat.jpg") #baca file dan simpan ke variabel "gambar"
cv2.imshow("Gambar Kucing", gambar) #imshow untuk menampilkan gambar

cv2.waitKey(0) # 0 untuk exit dengan menekan enter