import numpy as np
import cv2

I = cv2.imread("I04.jpg")
h = I.shape[0]
w = I.shape[1]
Ir = I[:,:,2]
Ig = I[:,:,1]
Ib = I[:,:,0]

# cap phat anh RGB moi
I_avg = np.zeros((h,w,3),dtype="uint8")

I_avg[:,:,2] = cv2.blur(Ir,(3,3))
I_avg[:,:,1] = cv2.blur(Ig,(3,3))
I_avg[:,:,0] = cv2.blur(Ib,(3,3))

cv2.imshow("Anh goc",I)
cv2.imshow("Anh trung binh cong",I_avg)
cv2.waitKey()