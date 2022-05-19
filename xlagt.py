import cv2
import numpy as np

Img = cv2.imread("CMTND02.jpg")
# cv2.imshow("Giao thong",Img)

# Chuyển ảnh gốc sang ảnh xám = cách tự động
# Img_Gray = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("Anh giao thong sau khi chuyen mau",Img_Gray)

# Chuyển ảnh gốc sang ảnh xám = cách thủ công
print("Chuyển ảnh sang grayscale bằng cách thủ công")
rows = Img.shape[0]
cols = Img.shape[1]
print(rows)
print(cols)

lg = np.zeros((rows,cols),dtype="uint8")
# Cách 1:
# for i in range(rows):
#     for j in range(cols):
#         # gray = 11 * r + 50 * g + 39 * b
#         # d = 11 * int(Img[i][j][2]) + 39 * int(Img[i][j][0]) + 50 * int(Img[i][j][1])
#         d = 11 * int(Img[i,j,2]) + 39 * int(Img[i,j,0]) + 50 * int(Img[i,j,1])
#         d = d // 100
#         lg[i][j] = d

# Cách 2:
def toGray(Img):
    lg = 0.11 * Img[:,:,2] + 0.5 * Img[:,:,1] + 0.39 * Img[:,:,0]
    lg = lg.astype(dtype="uint8")
    return lg
lg2 = toGray(Img)
cv2.imshow("Anh xam",lg2)
cv2.waitKey(0)