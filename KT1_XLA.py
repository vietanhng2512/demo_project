import cv2

I = cv2.imread("I04.jpg")
# 2. Hien thi kenh B cua anh
cv2.imshow("anh I", I[:, :, 0])

# 3. gia tri mau tai toa do y = 10, x=35
print("R = ", I[10][35][2])
print("G = ", I[10][35][1])
print("B = ", I[10][35][0])

# 3. resize anh
It = cv2.resize(I, (I.shape[0]//1, I.shape[1]//3))
cv2.imshow("anh resize", It)

# 4. Thay đổi độ rộng của I thành 400 pixel nhưng bảo toàn tỉ lệ của ảnh gốc được ảnh It. Hiển thị It.
h = I.shape[0]
w = I.shape[1]
print("Giá trị độ rộng: ",h)
print("Giá trị độ cao: ",w)
print("kich thuoc anh (wxh): {}x{}".format(h, w))

h_new = 400
w_new = (h * 400)//w
It_1 = cv2.resize(I, (h_new, w_new))
print("kich thuoc anh It_1:{}x{}".format(It_1.shape[0], It_1.shape[1]))
cv2.imshow("It_1", It_1)

# 5. chuyen anh gray
def toGray(I):
    Ig=0.11*I[:,:,2]+0.5*I[:,:,1]+0.39*I[:,:,0]
    Ig=Ig.astype(dtype='uint8')
    return Ig
Ig = toGray(I)
cv2.imshow("Ig gray", Ig)

# 6. Chuyển ảnh Ig sang ảnh nhị phân Ib với ngưỡng Otsu. Hiển thị giá trị ngưỡng Otsu tính được và hiển thị ảnh Ib.
ret, Ib = cv2.threshold(Ig, 0, 255, cv2.THRESH_OTSU)
print("nguong:", ret)
# cv2.imshow("Img binary", Ib)

# 7.  Chuyển đổi ảnh I sang biểu diễn mầu HSV được ảnh Ihsv. Hiển thị kênh V của ảnh Ihsv.
Ihsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
cv2.imshow("Kenh V cua anh", Ihsv[:, :, 2])

# 8. Hiển thị giá trị mức sáng của kênh V của ảnh Ihsv tại pixel có tọa độ dòng y=10, cột x=35 (tọa độ dòng, cột tính từ 0).
print("Gia tri muc sang:", Ihsv[10,35,2])

# 9 Chuyển đổi kênh S của ảnh Ihsv thành ảnh nhị phân bằng phương pháp Otsu và được ảnh Is. Hiển thị Is.
ret_1, Is = cv2.threshold(Ihsv[:, :, 1], 0, 255, cv2.THRESH_OTSU)
print("nguong:", ret_1)
cv2.imshow("Img binary", Is)

# 10. Gán kênh S của Ihsv là ma trận ảnh Is ở câu 8. Biến đổi ngược ảnh Ihsv về biểu diễn mầu BGR được ảnh
Ihsv[:, :, 1] = Is
I2 = cv2.cvtColor(Ihsv, cv2.COLOR_HSV2BGR)
cv2.imshow("I2", I2)

cv2.waitKey(0)