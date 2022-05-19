import cv2
import numpy as np
import matplotlib.pyplot as plt

# matplotlib inline
plt.rcParams['figure.figsize'] = [10, 8]
Img = cv2.imread("dark.jpg",0)
cv2.imshow("Anh",Img)

# Img_color_convert = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Anh chuyen doi",Img_color_convert)
#
# T,thresh = cv2.threshold(Img_color_convert, 0, 255, cv2.THRESH_OTSU)
# print("Ngưỡng",T)
# cv2.imshow("Anh theo nguong", thresh)

# using cv2.calcHist()
hist = cv2.calcHist(
    [Img],
    channels= [0],
    mask= None, # full image
    histSize= [256], # full scale
    ranges= [0,256]
)
plt.plot(hist)
plt.show()

# using numpy
h2 = np.histogram(Img.ravel(), bins= 256, range= [0, 256])
print(h2[0].shape)
plt.plot(h2[0])
plt.show()

def cal_his(Igray):
    row = Igray.shape[0]
    col = Igray.shape[1]

    mang_his = np.zeros(256, dtype='uint32')
    for i in range(0, row):
        for j in range(0, col):
            g = Igray[i][j]
            mang_his[g] = mang_his[g] + 1
    return mang_his

hist = cal_his(Img)
plt.plot(hist)
plt.show()
hist_eq = cv2.equalizeHist(Img)
plt.plot(hist_eq)
plt.show()
cv2.imshow("img hist", hist_eq)
cv2.waitKey(0)



