import cv2
import numpy as np
I = cv2.imread("clother1.jpg")
cv2.imshow('RGB mode for clothes: ',I)

# cv2.imshow('Kenh H',I[:,:,0])
# cv2.imshow('Kenh S',I[:,:,1])
# cv2.imshow('Kenh V',I[:,:,2])

# Hiển thị từng kênh r,g,b
cv2.imshow("Kenh R",I[:,:,2])
cv2.imshow("Kenh G",I[:,:,1])
cv2.imshow("Kenh B",I[:,:,0])

# Chuyển biểu diễn màu sang HSV
I_hsv = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)
# Hiển thị từng kênh h,s,v
cv2.imshow('HSV mode',I_hsv)
cv2.imshow('Kenh H',I[:,:,0])
cv2.imshow('Kenh S',I[:,:,1])
cv2.imshow('Kenh V',I[:,:,2])
# Chuyển từ hsv về rgb
I_2 = cv2.cvtColor(I_hsv,cv2.COLOR_BGR2HSV)
cv2.imshow('anh 2',I_2)

cv2.imwrite('pastel.hsv.bmp',I_hsv)
cv2.waitKey(None)



