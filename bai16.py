import cv2
import numpy as np
from matplotlib import pyplot as plt

def tinh_his(Igray):
    w = Igray.shape[1]
    h = Igray.shape[0]

    mang_hist = np.zeros(256,dtype="uint32")
    for i in range(h):
        for j in range(w):
            g = Igray[i][j]
            mang_hist[g] = mang_hist[g] + 1
    return mang_hist

I = cv2.imread("5.jpg")
Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
cv2.imshow("Anh goc",Igray)

hist = tinh_his(Igray)
plt.plot(hist)
plt.show()

Igray_eq = cv2.equalizeHist(Igray)
cv2.imshow("Anh can bang",Igray_eq)
hist_eq = tinh_his(Igray_eq)
plt.plot(hist_eq)
plt.show()
cv2.waitKey()