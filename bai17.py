import cv2
import numpy as np

def avg_filt(Igray,d):
    h = Igray.shape[0]
    w = Igray.shape[1]
    Igray_new = np.zeros((h,w),dtype="uint8") # cap phat 1 anh gray 8 bit
    for i in range(h):
        for j in range(w):
            g_sum = 0
            for ii in range(i - d, i + d+1):
                for jj in range(j - d, j + d+1):
                    if(ii >= 0) & (ii <= h-1) & (jj >= 0) & (jj <= w-1):
                        g_sum = g_sum + Igray[ii][jj]

        Igray_new[i][j] = g_sum//((2 * d+1)*(2 * d+1))
    return Igray_new

I = cv2.imread("I04.jpg")
h = I.shape[0]
w = I.shape[1]
Ir = I[:,:,2]
Ig = I[:,:,1]
Ib = I[:,:,0]
cv2.imshow("Anh goc", I)

# cap phat anh RGB moi
I_avg = np.zeros((h,w,3),dtype="uint8")

I_avg[:,:,2] = avg_filt(Ir,1)
I_avg[:,:,1] = avg_filt(Ig,1)
I_avg[:,:,0] = avg_filt(Ib,1)

cv2.imshow("Anh goc",I)
cv2.imshow("Anh trung binh cong smooth",I_avg)
cv2.waitKey()


