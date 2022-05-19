import cv2
import numpy as np

def gian_muc_xam(Igray):
    w = Igray.shape[1]
    h = Igray.shape[0]

    a = np.min(Igray)
    b = np.max(Igray)

    # chỉnh lại
    for i in range(h):
        for j in range(w):
            Igray[i,j] = (255 * int(Igray[i,j] - a))//(b - a)
    return a,b,Igray

I = cv2.imread('3.jpg')
cv2.imshow("RGB mode",I)

cv2.imshow("Kenh R",I[:,:,2])
cv2.imshow("Kenh G",I[:,:,1])
cv2.imshow("Kenh B",I[:,:,0])

rmin,rmax,I[:,:,2] = gian_muc_xam(I[:,:,2])
gmin,gmax,I[:,:,1] = gian_muc_xam(I[:,:,1])
bmin,bmax,I[:,:,0] = gian_muc_xam(I[:,:,0])

cv2.imshow("Kenh R moi",I[:,:,2]) # giãn mức xám kênh R
cv2.imshow("Kenh G moi",I[:,:,1]) # giãn mức xám kênh G
cv2.imshow("Kenh B moi",I[:,:,0]) # giãn mức xám kênh B

cv2.imshow("RGB moi",I)

cv2.waitKey()

# Luyện tập
# I = cv2.imread("3.jpg")
I = cv2.imread("I04.jpg")
cv2.imshow("RGB goc",I)
Ihsv = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)
vmin,vmax,Iv = gian_muc_xam(Ihsv[:,:,2])
cv2.imshow('V goc',Ihsv[:,:,2])
cv2.imshow('V moi',Iv)

# biến đổi ngược
Ihsv[:,:,2] = Iv
I_moi = cv2.cvtColor(Ihsv,cv2.COLOR_HSV2BGR)
cv2.imshow('RGB moi sau khi bien doi nguoc',I_moi)

cv2.waitKey()

# 0 < gama < 1
def hieu_chinh_gamma(Ig,gama):
    aG = np.zeros(256,dtype="uint8")

    # tạo bảng tra độ xám mới
    for g in range(256):
        t = g/255.0
        t1 = np.power(t,gama)
        g1 = int(255 * t1)
        aG[g] = g1

    h = I.shape[0]
    w = I.shape[1]

    # duyệt từng điểm ảnh
    for i in range(h):
        for j in range(w):
            g = Ig[i,j]
            I[i,j] = aG[g]
    return Ig

I = cv2.imread("dark.jpg")
gama = 0.7
Ir = hieu_chinh_gamma(I[:,:,2],gama) # giãn mức xám kênh R
Ig = hieu_chinh_gamma(I[:,:,1],gama) # giãn mức xám kênh G
Ib = hieu_chinh_gamma(I[:,:,0],gama) # giãn mức xám kênh B

# hiển thị Ir,Ig,Ib
cv2.imshow('R gamma',Ir)
cv2.imshow('G gamma',Ig)
cv2.imshow('B gamma',Ib)

I[:,:,2] = Ir
I[:,:,1] = Ig
I[:,:,0] = Ib

cv2.imshow('RGB sau gamma',I)

cv2.waitKey()
