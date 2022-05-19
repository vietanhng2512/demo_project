import cv2
import numpy as np
import matplotlib.pyplot as plt

# a
I = cv2.imread("hoa1.jpg")
Ihsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
cv2.imshow("Kenh H cua anh Ihsv", Ihsv[:,:,0])

print(sep = "\n")
# b
hist = cv2.calcHist(
    [Ihsv],
    channels = [2],
    mask = None,
    histSize=[255],
    ranges = [0,255]
)
print("Kênh V được cân bằng: \n", hist)
plt.plot(hist)
plt.show()

print(sep = "\n")
# c
Ihsv[:,:,2] = cv2.equalizeHist(Ihsv[:,:,2])

I_2 = cv2.cvtColor(Ihsv, cv2.COLOR_HSV2BGR)
cv2.imshow("Anh chuyen",I_2)

# d
cv2.imwrite("hoa1.png",I_2)
cv2.imshow("Anh I2 sau khi chuyen mau",I_2)

# e
h = I.shape[0]
w = I.shape[1]
I_resize = cv2.resize(I, (h//1, w//2))
cv2.imshow("Size anh sau khi giam",I_resize)
cv2.waitKey(0)




