import cv2 as cv
img = cv.imread("download.jpg", 0)
print(img)
cv.imshow("Anh: ",img)
res = cv.resize(img,None,fx=2,fy=2,interpolation=cv.INTER_LINEAR)
cv.imwrite("test.jpg", img)
cv.waitKey(0)