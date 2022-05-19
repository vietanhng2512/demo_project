import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# matplotlib inline
plt.rcParams['figure.figsize'] = [8, 6]

img = cv2.imread("pepper.jpg")
plt.imshow(img[:,:,::-1])
plt.axis("off")
print(img.shape)
plt.show()

# print("abc")
def rgb_filter(img, kernel, mode= "full", boundary= "fill"):
    b, g, r = cv2.split(img)
    b = signal.correlate2d(b, kernel, mode= mode, boundary= boundary)
    g = signal.correlate2d(g, kernel, mode= mode, boundary= boundary)
    r = signal.correlate2d(r, kernel, mode= mode, boundary= boundary)
    output = cv2.merge([b, g, r])
    return output

# make a kernel
kernel = np.zeros((21, 21), dtype= "uint8")
kernel[10, 10] = 1

# do cross corrlation
f_img = rgb_filter(img, kernel, mode= "full", boundary= "fill")

# check the size of image
print("Original shape: ",img.shape)
print("Filtered shape: ",f_img.shape)

# view the image
plt.subplot(1, 2, 1)
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(f_img[:,:,::-1])
plt.axis("off")
plt.show()

print(sep= "\n")
# do cross corrlation
f_img = rgb_filter(img, kernel, mode= "full", boundary= "wrap")

# check the size of image
print("Original shape: ",img.shape)
print("Filtered shape: ",f_img.shape)

# view the image
plt.subplot(1, 2, 1)
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(f_img[:,:,::-1])
plt.axis("off")
plt.show()

print(sep= "\n")
# symmetry boundary
f_img = rgb_filter(img, kernel, mode= "full", boundary= "symm")

# check the size of image
print("Original shape: ",img.shape)
print("Filtered shape: ",f_img.shape)

# view the image
plt.subplot(1, 2, 1)
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(f_img[:,:,::-1])
plt.axis("off")
plt.show()

print(sep= "\n")
# valid filtering
f_img = rgb_filter(img, kernel, mode= "valid", boundary= "fill")

# check the size of image
print("Original shape: ",img.shape)
print("Filtered shape: ",f_img.shape) ## should decrease

# view the image
plt.subplot(1, 2, 1)
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(f_img[:,:,::-1])
plt.axis("off")
plt.show()

print(sep= "\n")
# same filtering
f_img = rgb_filter(img, kernel, mode= "same", boundary= "fill")

# check the size of image
print("Original shape: ",img.shape)
print("Filtered shape: ",f_img.shape) ## equal

# view the image
plt.subplot(1, 2, 1)
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(f_img[:,:,::-1])
plt.axis("off")
plt.show()

print(sep= "\n")
# Using opencv
f_img = cv2.filter2D(img, -1,kernel, cv2.BORDER_DEFAULT)

# check the size of image
print("Original shape: ",img.shape)
print("Filtered shape: ",f_img.shape)

# view the image
plt.subplot(1, 2, 1)
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(f_img[:,:,::-1])
plt.axis("off")
plt.show()

# Averaging
kernel = 1/9 * np.array([
               [1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]
])

# averaging for a smoothing effect
f_img = cv2.filter2D(img, -1, kernel)
plt.subplot(1, 2, 1)
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(f_img[:,:,::-1])
plt.axis("off")
plt.show()

# Gaussian Filtering
kernel = 1/12 * np.array([
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]
])

# averaging for smoothing effect
f_img = cv2.filter2D(img, -1, kernel)
plt.subplot(1, 2, 1)
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(f_img[:,:,::-1])
plt.axis("off")
plt.show()

# or using GaussianBlur
f_img = cv2.GaussianBlur(img, (3, 3), 0)
plt.subplot(1, 2, 1)
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(f_img[:,:,::-1])
plt.axis("off")
plt.show()

f_img = cv2.GaussianBlur(img, (5, 5), 0) # image, kernel size, sigma
plt.subplot(1, 2, 1)
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(f_img[:,:,::-1])
plt.axis("off")
plt.show()

f_img = cv2.GaussianBlur(img, (21, 21), 0) # image, kernel size, sigma
plt.subplot(1, 2, 1)
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(f_img[:,:,::-1])
plt.axis("off")
plt.show()

f_img = cv2.GaussianBlur(img, (21, 21), 20) # image, kernel size, sigma
plt.subplot(1, 2, 1)
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(f_img[:,:,::-1])
plt.axis("off")
plt.show()

# Median filtering
f_img = cv2.medianBlur(img, 5) # image, kernel size, sigma
plt.subplot(1, 2, 1)
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(f_img[:,:,::-1])
plt.axis("off")
plt.show()

f_img = cv2.medianBlur(img, 45) # image, kernel size
plt.subplot(1, 2, 1)
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(f_img[:,:,::-1])
plt.axis("off")
plt.show()

# Unsharp masking
# dst=α⋅src1+β⋅src2+γ
#
# Where src2 is a blurred image
blur = cv2.GaussianBlur(img, (7,7), 30)
f_img = cv2.addWeighted(img, 1.7, blur, -0.5, 0) # f_img = image * 1.7 - 0.5 * blur + 0
plt.subplot(1, 3, 1)
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.title("original")
plt.show()

plt.subplot(1, 3, 2)
plt.imshow(f_img[:,:,::-1])
plt.axis("off")
plt.title("blur(unsharp mark)")
plt.show()

plt.subplot(1, 3, 3)
plt.imshow(f_img[:,:,::-1])
plt.axis("off")
plt.title("sharpen")
plt.show()