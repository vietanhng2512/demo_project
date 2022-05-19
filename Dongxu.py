import cv2
import numpy as np

img = cv2.imread("dongxu.jpg")
img_coins = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_coins, (5,5), 0)
cv2.imshow("Image",img)
cv2.imshow("Coins blur",img_blur)

# (T,thresh) = cv2.threshold(img_blur,155,255,cv2.THRESH_BINARY)
# print("Ngưỡng lần 1:",T)
# cv2.imshow("Thresh Binary",thresh)
#
# thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                cv2.THRESH_BINARY_INV, 5, 3)
# cv2.imshow("Mean thresh",thresh)
#
# thresh_gaussian = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                         cv2.THRESH_BINARY_INV, 5, 3)
# cv2.imshow("Gaussian thresh",thresh_gaussian)
# cv2.waitKey(0)

# T,thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU)
# print("Ngưỡng: ",T)
# cv2.imshow("Thresholded",thresh)
# print("Otsu thresholding value: {}".format(T))
# cv2.waitKey(0)

(T,threshInv) = cv2.threshold(img_blur, 155, 255, cv2.THRESH_BINARY_INV)
print("Ngưỡng lần 2:",T)
cv2.imshow("Thresh Binary Invert",threshInv)
cv2.imshow("Coins", cv2.bitwise_and(img_coins, img_coins, mask= threshInv))

# Loại bỏ từng mảng đen ra khỏi đồng xu
threshFilled = threshInv.copy()
mask = np.zeros((threshInv.shape[0]+2, threshInv.shape[1]+2), dtype="uint8")
cv2.floodFill(threshFilled, mask, (0,0), 255)
threshFilled = cv2.bitwise_not(threshFilled)
threshFilled = cv2.bitwise_or(threshFilled, threshInv)

cv2.imshow("Threshold binary filled",threshFilled)
cv2.imshow("Coins Convert", cv2.bitwise_and(img_coins, img_coins, mask= threshFilled))
cv2.waitKey(0)





