import cv2
import matplotlib.pyplot as plt
import numpy as np

# Đọc vào bong1.jpg được biến ma trận ảnh I
I = cv2.imread("bong1.jpg")
cv2.imshow("Anh goc", I)

# 25a. Chuyển ảnh sang biểu diễn HSV được ma trận Ihsv. Hiển thị kênh H của Ihsv.
Ihsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
cv2.imshow("Kenh H cua anh Ihsv", Ihsv[:,:,0])

# 25b. Cân bằng histogram của kênh V của Ihsv. Hiển thị kênh V được cân bằng.
Ihsv_hist = cv2.calcHist(
    [Ihsv],
    channels = [2],
    mask = None,
    histSize = [255],
    ranges = [0,255],
)
print("Kênh V được cân bằng: \n", Ihsv_hist)
plt.plot(Ihsv_hist)
plt.show()
print("s")

# 25c. Thay đổi kênh V của Ihsv thành kênh V đã cân bằng. Chuyển Ihsv về biểu diễn RGB được ảnh I2.
# Hiển thị I2.
Ihsv[:,:,2] = cv2.equalizeHist(Ihsv[:,:,2])
I_2 = cv2.cvtColor(Ihsv, cv2.COLOR_HSV2BGR)
cv2.imshow("Anh chuyen doi mau", I_2)

# 25d. Ghi ma trận ảnh I2 thành file ảnh hoa1.png
cv2.imwrite("hoa1.jpg",I_2)

#25e. Giảm size ảnh xuống 1/4. Hiển thị ảnh đã giảm size.
h = I.shape[0]
w = I.shape[1]
I_resize = cv2.resize(I, (h//1, w//4))
cv2.imshow("Anh sau khi giam size", I_resize)

print(sep = "\n")
#25g. Làm trơn ảnh kênh H của Ihsv theo bộ lọc trung bình cộng, kích thước cửa sổ lân cận là 7x7 được ảnh Ih. Hiển thị ảnh Ih.
matran_trongso = np.ones((7,7), np.float32)/49
print(matran_trongso)
Ih = cv2.filter2D(Ihsv[:,:,0], -1, matran_trongso)
cv2.imshow("Loc trung binh cong 7 x 7", Ih)

#25h. Nhị phân hóa anh Ih theo ngưỡng Otsu được ảnh nhị phân Ib. Hiển thị ảnh Ib.
T, Ib = cv2.threshold(Ih, 0, 255, cv2.THRESH_OTSU)
print("Ngưỡng: ", T)
cv2.imshow("Anh Ib theo nguong", Ib)

# 25o. Chuyển ảnh I sang ảnh grayscale theo công thức biến đổi bộ mầu (r,g,b) về mức xám=0.39*r+0.50*g+0.11*b, được ảnh Ig. Hiển thị ảnh Ig.
def toGray(I):
    Ig = 0.11 * I[:,:,0] + 0.50 * I[:,:,1] + 0.39 * I[:,:,2]
    Ig = Ig.astype(dtype = "uint8")
    return Ig

Ig = toGray(I)
cv2.imshow("Anh Ig", Ig)

cv2.waitKey(0)