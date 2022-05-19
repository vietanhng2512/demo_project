import cv2
import numpy as np
from matplotlib import pyplot as plt

def tinh_his(Igray):
    w = Igray.shape[1]
    h = Igray.shape[0]

    mang_hist = np.zeros(256,dtype='uint32')
    for i in range(h):
        for j in range(w):
            g = Igray[i][j]
            mang_hist[g] = mang_hist[g] + 1
    return mang_hist

I = cv2.imread('I04.jpg')
cv2.imshow('I04 goc',I)
Ir = I[:,:,2]
Ig = I[:,:,1]
Ib = I[:,:,0]

hist_r = tinh_his(Ir)
plt.plot(hist_r)
plt.show()
cv2.waitKey()




