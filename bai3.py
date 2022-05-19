import cv2
import numpy as np

I = cv2.imread("Traffic.jpg")
cv2.imshow("anh goc",I)

#jpg -> png
#a1. dễ nhất
cv2.imwrite('CMTND02.png',I)
I_png = cv2.imread("CMTND02.png")
cv2.imshow("PNG",I_png)
cv2.waitKey()
