import cv2 as cv2
import numpy as np

# print("Đọc ảnh")
Img = cv2.imread("Traffic.jpg")
# cv2.imshow("Anh goc",Img)
print(Img)
exit()
# print("Kenh B:", Img[:, : , 0])
# print("Kenh G:", Img[:, : , 1])
# print("Kenh R:", Img[:, : , 2])
# cv2.imshow("Kenh red",Img[:,:,2])
# cv2.imshow("Kenh green",Img[:,:,1])
# cv2.imshow("Kenh blue",Img[:,:,0])
# cv2.imwrite("gt.jpg",Img)

cv2.waitKey(0)




