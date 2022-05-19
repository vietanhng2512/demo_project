import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"Đọc ảnh I chuyển thành ảnh xám Ig"
I = cv.imread("dongxu.jpg")
Igray = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
cv.imshow("Image gray", Igray)

"Xác định ma trận gradient theo 2 hướng x và y sử dụng Sobel"
Sobel_x = cv.Sobel(Igray, cv.CV_64F, 1, 0, 3)
Sobel_y = cv.Sobel(Igray, cv.CV_64F, 0, 1, 3)

"Hiển thị 2 ma trận"
plt.subplot(2, 2, 1), plt.imshow(Sobel_x, cmap = "gray")
plt.title(Sobel_x), plt.xticks([]), plt.yticks([])
plt.show()

"Xác định ảnh biên của ảnh Igray sử dụng toán tử Sobel và hiển thị kết quả"
Ig_Sobel = np.sqrt(Sobel_x**2 + Sobel_y**2)
cv.imshow("Ig_Sobel",Ig_Sobel)

"Hiển thị các độ xám của cửa sổ lân cận 3 x 3 của điểm ảnh gray(y = 179, x = 123)"
height = Igray.shape[0]
width = Igray.shape[1]
y = 319
x = 279
for k in range(-1, 2):
    for l in range(-1, 2):
        if((y + k) >= 0) & ((y + k) <= height - 1) & ((x + l) <= width - 1):
            print("Các mức độ xám của cửa sổ lân cận 3 x 3 điểm ảnh gray(y = 179,x = 123)",
                  Igray[y + k],[x + l])

"Xác định ảnh biên của ảnh Igray sử dụng phương pháp Canny và hiển thị kết quả"
I_Canny = cv.Canny(Igray, 0, 255)
cv.imshow("Image Canny",I_Canny)

cv.waitKey(0)




