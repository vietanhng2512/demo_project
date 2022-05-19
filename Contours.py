import cv2

I = cv2.imread("contour1.jpg")
# cv2.imshow("contour1.jpg", I)

height = I.shape[0]
width = I.shape[1]
I_1 = cv2.resize(I, (width * 2, height * 2))
# cv2.imshow("contour resize", I_1)

Ig = cv2.cvtColor(I_1, cv2.COLOR_BGR2GRAY)
thresh, Ib = cv2.threshold(Ig, 0, 255, cv2.THRESH_OTSU)

contours,_ = cv2.findContours(Ib, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(I_1, contours, -1, (0, 0, 255), 2) # đỏ
cv2.imshow("Ve contour", I_1)

# Tính diện tích contour
max_area = 0.0
contour_max = []
for cnt in contours:
    if max_area <= cv2.contourArea(cnt):
        max_area = cv2.contourArea(cnt)
        contour_max = cnt
cv2.drawContours(I_1, [contour_max], -1, (0, 255, 0), 2)
cv2.imshow("Dien tich contour", I_1) # Xanh lá

# Tính chu vi contours
max_per = 0.0
for cnt in contours:
    if max_per <= cv2.arcLength(cnt, True):
        max_per = cv2.arcLength(cnt, True)
        contour_max = cnt
cv2.drawContours(I_1, [contour_max], -1, (0, 255, 255), 2)
cv2.imshow("chu vi contour", I_1) # Vàng

# Tìm contour có tỉ số giữa diện tích và bình phương chu vi là lớn nhất
max_ts = 0.0
for cnt in contours:
    tiso_S_CV = (cv2.contourArea(cnt)) / ((cv2.arcLength(cnt, True))**2)
    if max_ts <= tiso_S_CV:
        max_ts = tiso_S_CV
        contour_max = cnt
cv2.drawContours(I_1, [contour_max], -1, (255, 255, 0), 2)
cv2.imshow("Ti so dien tich va chu vi contour", I_1) # Xanh

cv2.waitKey(0)