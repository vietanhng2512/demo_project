import cv2
import numpy as np
import random

I = cv2.imread("I04.jpg")
matran_trongso = np.ones((5, 5), np.float32)/25
print(matran_trongso)
I_1 = cv2.filter2D(I, -1, matran_trongso)
cv2.imshow("Loc trung binh cong 5 x 5", I_1)

# Thay đổi ma trận matran_trongso để xem kết quả
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
cv2.imshow("loc trung binh co trong so",I_2)
cv2.waitKey()

# Khối lệnh sau để làm gì?
#
# for i in range(7): for j in range(7): matran_trongso[i][j]=matran_trongso[i][j]/s
#
# Để lọc lấy trung bình trọng số trong khoảng 7x7




