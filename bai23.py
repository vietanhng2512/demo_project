import cv2
import numpy as np

# a
I = cv2.imread("anh5.jpg")

# b
Ig = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
cv2.imshow("Anh xam Ig",Ig)

# c
T, thresh = cv2.threshold(Ig, 0, 255, cv2.THRESH_OTSU)
print("Ngưỡng: ", T)
cv2.imshow("Anh nhi phan nen den theo Otsu", thresh)

# l
Ihsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
Is = Ihsv[:,:,1]
cv2.imshow("Anh kenh S cua Ihsv", Is)

# o
def toGray(I):
    Ig = 0.11 * I[:,:,0] + 0.50 * I[:,:,1] + 0.39 * I[:,:,2]
    Ig = Ig.astype(dtype = "uint8")
    return Ig
Ig = toGray(I)
cv2.imshow("Anh grayscale", Ig)

cv2.waitKey(0)