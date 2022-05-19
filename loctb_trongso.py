import cv2
import numpy as np
import random

I = cv2.imread("I04.jpg")
matran_trongso = np.zeros((7,7), dtype = "float32")
s = 0.0
for i in range(7):
    for j in range(7):
        matran_trongso[i][j] = random.random()
        s = s + matran_trongso[i][j]
for i in range(7):
    for j in range(7):
        matran_trongso[i][j] = matran_trongso[i][j] / s

print(matran_trongso)
I_2 = cv2.filter2D(I, -1, matran_trongso)
cv2.imshow("loc trung binh co trong so", I_2)

def trung_binh_trong_so(Igray, d, ma_tran_trong_so):
    h = Igray.shape[0]
    w = Igray.shape[1]
    Igray_new = np.zeros((h,w), dtype = "uint8") # cap phat 1 anh gray 8 bit

    for i in range(h):
        for j in range(w):
            g_sum = 0
            for k in range(-d, d + 1):
                for l in range(-d, d + 1):
                    if(i + k >= 0) & (i + k <= h - 1) & (j + l >= 0) & (j + l <= w - 1):
                        g_sum = g_sum + float(Igray[i + k][j + l]) * \
                                float(ma_tran_trong_so[d + k][d + l])
                        Igray_new[i][j] = int(g_sum)
                        return Igray_new

def avg_filt(Igray, d):
    h = Igray.shape[0]
    w = Igray.shape[1]
    Igray_new = np.zeros((h, w), dtype= "uint8") # cap phat 1 anh gray 8 bit
    for i in range(h):
        for j in range(w):
            g_sum = 0
            for ii in range(i - d, i + d + 1):
                for jj in range(j - d, j + d + 1):
                    if(ii >= 0) & (ii <= h - 1) & (jj >= 0) & (jj <= w - 1):
                        g_sum = g_sum + Igray[ii][jj]
                        Igray_new[i][j] = g_sum // ((2 * d + 1) * (2 * d + 1))
                        return Igray_new

I = cv2.imread("I04.jpg")
h = I.shape[0]
w = I.shape[1]
Ir = I[:,:,2]
Ig = I[:,:,1]
Ib = I[:,:,0]

# cap phat anh RGB moi
I2 = np.zeros((h,w,3), dtype = "uint8")

ma_tran_trong_so = np.zeros((7, 7), dtype = "float32")
s = 0.0
for i in range(7):
    for j in range(7):
        ma_tran_trong_so[i][j] = random.random()
        s = s + ma_tran_trong_so[i][j]
for i in range(7):
    for j in range(7):
        ma_tran_trong_so[i][j] = ma_tran_trong_so[i][j] / s
print(ma_tran_trong_so)
I2[:,:,2]=trung_binh_trong_so(Ir,3,ma_tran_trong_so)
I2[:,:,1]=trung_binh_trong_so(Ig,3,ma_tran_trong_so)
I2[:,:,0]=trung_binh_trong_so(Ib,3,ma_tran_trong_so)

cv2.imshow("Anh goc",I)
cv2.imshow("Anh smooth trung binh trong so",I2)
I2[:,:,2] = cv2.blur(Ir,(3, 3))
I2[:,:,1] = cv2.blur(Ig,(3, 3))
I2[:,:,0] = cv2.blur(Ib,(3, 3))
I_2 = cv2.filter2D(I, -1, matran_trongso)
cv2.imshow("Loc trung binh co trong so voi filter 2D",I_2)
cv2.imshow("Loc trung binh voi blur tu code",I2)

I_avg = np.zeros((h,w,3), dtype= "uint8")
I_avg[:,:,2] = cv2.blur(Ir, (3, 3))
I_avg[:,:,1] = cv2.blur(Ig, (3, 3))
I_avg[:,:,0] = cv2.blur(Ib, (3, 3))
cv2.imshow("Loc trung binh trong so", I_2)
cv2.imshow("Loc trung binh cong", I_avg)
I_1 = cv2.filter2D(I, -1, matran_trongso)
cv2.imshow("Loc trung binh cong 5 x 5", I_1)
cv2.waitKey()
