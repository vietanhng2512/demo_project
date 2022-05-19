import cv2
import numpy as np

I = cv2.imread("./I04.jpg")
cv2.imshow("Anh",I[:,:,0])

print("B = ",I[10,35,0])
print("G = ",I[10,35,1])
print("R = ",I[10,35,2])

rows = I.shape[0]
cols = I.shape[1]
It = cv2.resize(I,(rows//1,cols//3))
cv2.imshow("Anh thu gon",It)

def toGray(I):
    Ig = 0.11 * I[:,:,0] + 0.5 * I[:,:,1] + 0.39 * I[:,:,2]
    Ig = Ig.astype(dtype="uint8")
    return Ig
Ig = toGray(I)
cv2.imshow("Ig gray",Ig)

ret,Ib = cv2.threshold(Ig,0,255,cv2.THRESH_OTSU)
print("Giá trị ngưỡng Otsu của ảnh Ib: ",ret)
cv2.imshow("Anh nhi phan Ib",Ib)

Ihsv = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)
print(sep="\n")
print("Kênh V của ảnh Ihsv: \n",Ihsv[:,:,2])
print("Giá trị mức sáng của kênh V = ",Ihsv[10][35][2])

ret_1, Is = cv2.threshold(Ihsv[:,:,1],0,255,cv2.THRESH_OTSU)
print("Ngưỡng: ",ret_1)
cv2.imshow("Anh nhi phan Is",Is)

Ihsv[:,:,1] = Is
I2 = cv2.cvtColor(Ihsv,cv2.COLOR_HSV2BGR)
cv2.imshow("Anh I2",I2)

cv2.waitKey(0)


