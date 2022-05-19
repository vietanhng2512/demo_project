import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 8]
Img_xray = cv2.imread("xray.jpeg",0)
# cv2.imshow("Anh",Img_xray)
plt.imshow(Img_xray, cmap="gray")
plt.show()

# Image negatives
# Ta có thể chuyển từ ảnh bên trái sang bên phải ( đổi đen thành trắng và trắng thành đen 😃)
negation = 255 - Img_xray
plt.imshow(negation, cmap="gray")
plt.show()

# plot a quantization image of k level
def draw_quantization_img(levels, height = 32):
    # convert it to an image
    Img = [levels] * height
    img = np.array(Img)
    plt.imshow(Img, "gray")
    #plt.axis('off')
    return img

gray_256 = list(range(0, 256, 1))
draw_quantization_img(gray_256)
plt.show()

gray_64 = list(range(0, 256, 4))
print(len(gray_64))
draw_quantization_img(gray_64, height= 8)
plt.title("64", loc= "left")
plt.show()

gray_32 = list(range(0, 256, 8))
print(len(gray_32))
draw_quantization_img(gray_32, height= 4)
plt.title("32", loc= "left")
plt.show()

gray_16 = list(range(0, 256, 16))
print(len(gray_16))
draw_quantization_img(gray_16, height= 2)
plt.title("16", loc= "left")
plt.show()

gray_8 = list(range(0, 256, 32))
print(len(gray_8))
draw_quantization_img(gray_8, height= 2)
plt.title("8", loc= "left")
plt.show()

# Gamma
# vout=vγin
# vin  : độ sáng thực tế (actual luminance value)
#
# vout : độ sáng cảm nhận (output luminance value)
def adjust_gamma(inlevels, gamma= 1.0, debug= True):
    out = [l ** gamma for l in inlevels]
    max_out = max(out)
    out = [int(l/max_out * 256) for l in out]
    print(out)
    return out

# Không điều chỉnh gamma(no adjustment)
draw_quantization_img(
    # Cho dải màu xám 16 mức, với gamma = 1 thì nó sẽ giữ nguyên như sau:
    adjust_gamma(gray_16, gamma= 1),
    height = 2
)
plt.title("[0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 256]", loc="left")
plt.show()

# Với gamma > 1, ta thấy nếu giá trị đầu vào là tối (thấp) thì đẩu ra là tối hơn,
# ví dụ đầu vào là giá trị 17 thì đầu ra là giá trị 0:
# with gamma > 1, the levels are shifted toward to the darker end of the spectrum
draw_quantization_img(
    adjust_gamma(gray_16, gamma= 2.2),
    height = 2
)
plt.title("[0, 0, 3, 7, 13, 22, 34, 47, 64, 83, 104, 129, 156, 186, 219, 256]", loc= "left")
plt.show()

# Với gamma < 1, các giá trị đầu vào là thấp thì đẩu ra sáng hơn,
# ví dụ đầu vào là 17 đầu ra là giá trị 74:
# with gamma < 1, lighter
draw_quantization_img(
    adjust_gamma(gray_16, gamma= 1/ 2.2, debug= "True"),
    height = 2
)
plt.title("[0, 74, 102, 123, 140, 155, 168, 181, 192, 202, 212, 222, 231, 239, 248, 256]", loc= "left")
plt.show()

# Logarithmic graph
x = np.arange(0, 1.2, 0.01)

g1 = np.power(x, 1)
g2 = np.power(x, 2.2) # gamma = 2.2
g3 = np.power(x, 5) # gamma = 5
g4 = np.power(x, 1/2.2)
g5 = np.power(x, 1/5)

fig, ax = plt.subplots()
ax.plot(x, g1, "r", label = "Linear")
ax.plot(x, g2, "b--", label = "gamma = 2.2")
ax.plot(x, g3, "b:", label = "gamma = 5")
ax.plot(x, g4, "g+", label = "gamma = 1/2.2")
ax.plot(x, g5, "g.", label = "gamma = 1/5")

ax.set_xlim([0, 1.2])
ax.set_ylim([0, 1.2])
legend = ax.legend(loc= "upper left", fontsize= "x-large")

plt.rcParams['figure.figsize'] = [8, 8]
plt.show()

# Sử dụng gamma để điều chỉnh độ tương phản
# 1. Độ tương phản thấp (Low exposure)
# Using gamma to adjust contrast
# 1. Low exposure
low_contrast = cv2.imread("dark.jpg")
plt.imshow(low_contrast[:,:,::-1])
plt.show()

def adjust_image_gamma(Img_xray, gamma = 1.0):
    img = np.power(Img_xray, gamma)
    max_val = np.max(img.ravel())
    img = img/max_val * 255
    img = img.astype(int)
    return img

low_adjust = adjust_image_gamma(low_contrast, 0.45)
plt.imshow(low_adjust[:,:,::-1])
plt.show()

# Nếu như gamma quá nhỏ thì sao ?
# what if gama if too low
low_adjusted = adjust_image_gamma(low_contrast, 0.1)
plt.imshow(low_adjusted[:,:,::-1])
plt.show()

# Ta thử đo thời gian với hàm adjust_image_gamma:
# timeit
# low_adjusted = adjust_image_gamma(low_contrast,0.1)
# print("Thời gian:",low_adjusted)

# Nó rất tốn thời gian, có cách nào nhanh hơn không ?. Thật ra là có 😃. Nó sẽ như sau:
# faster way to compute
# reference: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def adjust_image_gamma_lookuptable(img, gamma = 1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    table = np.array([((i/ 255.0) ** gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    # Ý tưởng ở đây là dùng bảng chuyển look-up table (LUT) sử dụng hàm cv2.LUT().
    return cv2.LUT(img, table)

low_adjust = adjust_image_gamma_lookuptable(low_contrast, 0.45)
plt.imshow(low_adjust[:,:,::-1])
plt.show()

# Thời gian chạy với hàm này sẽ là :
# timeit
# adjust_image_gamma_lookuptable(low_contrast, 0.45)
# print("Thời gian: ",adjust_image_gamma_lookuptable(low_contrast))

# 2. Độ tương phản cao (Overexposure)
# Với trường hợp này thì ta cần giảm độ sáng xuống, hay tăng độ tối. Vì vậy, sử dụng gamma > 1:
# Overexposure
high_contrast = cv2.imread("high-exposure.jpg")
plt.imshow(high_contrast[:,:,::-1])
plt.show()

# Ở đây sử dụng gamma = 4.
adjusted_high = adjust_image_gamma_lookuptable(high_contrast,4)
plt.imshow(adjusted_high[:,:,::-1])
plt.show()

# 2.3. Correct using pixel transform
# Reference: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
#
# Dùng phép toán nhân và cộng  g(x)=αf(x)+β
#
# α  và  β  còn được gọi là tham số gain và bias, hoặc tham số để điều chỉnh contrast (độ tương phản) và brightness (độ sáng)
#
# Với ảnh số:  g(i,j)=α⋅f(i,j)+β
def pixel_transform(img, alpha = 1.0, beta = 0):
    """
    out[pixel] = alpha * image[pixel] + beta
    """
    output = np.zeros(img.shape, img.dtype)
    h, w, ch = img.shape
    for y in range(h):
        for x in range(w):
            for c in range(ch):
                output[y, x, c] = np.clip(alpha * img[y, x, c] + beta, 0, 255)

    return output

transformed_high = pixel_transform(high_contrast, 0.5, 20)
plt.imshow(transformed_high[:,:,::-1])
plt.show()

# Có một cách dùng thư viện của opencv để nhanh hơn:
# anyway, a faster
transformed_high = cv2.convertScaleAbs(high_contrast, 20, 0.5)
plt.imshow(transformed_high[:,:,::-1])
plt.show()

# compare time
# pixel_transform(high_contrast, 0.5, 20)
# print(pixel_transform(high_contrast))

# time = % timeitcv2.convertScaleAbs(high_contrast, 20, 0.5)
# print(time)

# 2.4. Point operations for combining images
# 2.4.1. Image averaging for noise reduction
origin_img = cv2.imread("lena.jpg",0)
plt.imshow(origin_img, cmap= "gray")
plt.show()

# now make several image from noisy
def generate_noise_img(img, mean = 0, sigma = 0):
    gaussian = np.random.normal(mean, sigma, img.shape)
    noisy_img = img + gaussian
    return noisy_img

def make_batch(img, num_output = 4):
    sigmas = np.random.rand(num_output) * 2
    #print(sigmas)
    noisy_imgs = [generate_noise_img(img, sigma = sigma) for sigma in sigmas]
    return noisy_imgs

# make 4 images
noisy_Imgs = make_batch(origin_img, num_output= 4)
noisy_images = np.array(noisy_Imgs)
denoised = np.mean(noisy_images, axis= 0)
plt.imshow(denoised, cmap= "gray")
plt.show()

# Combining images
# 1. Combination of different exposures for high-dynamic range imaging
# Ta có 4 ảnh với 4 độ sáng khác nhau:
# 2.4.2. Combination of different exposures for high-dynamic range imaging
hdr_1 = cv2.imread("./img_hdr/hdr1.jpeg")
hdr_2 = cv2.imread("./img_hdr/hdr2.jpeg")
hdr_3 = cv2.imread("./img_hdr/hdr3.jpeg")
hdr_4 = cv2.imread("./img_hdr/hdr4.jpeg")

Stack_img = np.stack([hdr_1,hdr_2,hdr_3,hdr_4], axis = 0)
print(Stack_img.shape)

plt.rcParams['figure.figsize'] = [12, 8]
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(Stack_img[i][:,:,::-1])
plt.show()

# Ta sẽ kết hợp tính giá trị trung bình để tạo ra 1 ảnh mới đẹp hơn.
# Averaging to enhance contrast
hdr = np.mean(Stack_img, axis = 0)
HDR = hdr.astype(int)
plt.rcParams['figure.figsize'] = [8, 8]
plt.imshow(HDR[:,:,::-1])
plt.show()

# Đối với ai tò mò
# Reference: https://www.learnopencv.com/high-dynamic-range-hdr-imaging-using-opencv-cpp-python/
#
# Bao gồm:
#
# Căn chỉnh sử dụng AlignMTB
# Thuật toán chuyển đổi tất cả các hình ảnh thành bitmap ngưỡng trung bình (MTB)
# Một MTB cho hình ảnh được tính bằng cách gán giá trị 1 cho pixel sáng hơn độ chói trung bình và 0 nếu khác
# An MTB is invariant to the exposure time.
# Therefore, the MTBs can be aligned without requiring us to specifying the exposure time
alignMTB = cv2.createAlignMTB()
alignMTB.process(Stack_img, Stack_img)
# Obtain Camera Response Function (CRF)
# exposure time of each image, known from metadata
times = np.array([1/ 30.0, 0.25, 2.5, 15.0], dtype= np.float32)
calibrateDebevec = cv2.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(Stack_img, times)

mergeDebevec = cv2.createMergeDebevec()
hdrDebevec = mergeDebevec.process(Stack_img, times, responseDebevec)
plt.imshow(hdrDebevec[:,:,::-1])
plt.show()

# Tone mapping = Converting a HDR image to an 8-bit per channel image
# using HDR, the relative brightness information was recovered
# we need to convert the information as a 24-bit image for display
# Common parameters of the different tone mapping algorithms
# 1. gamma. For gamma correction; gamma < 1 darkens the image; gamma >1 brightens the image
# 2. saturation: to increase or decrease the amount of saturation. Higher -> colors are richer and more intense
#     Closer to zero, colors fade away to grayscale
# 3. contrast: controls the contrast. = log(maxPixelvalue/minPixelvalue)

# Opencv implements 4 tone mapping

# Tonemap using Drago's method to obtain 24-bit color image
tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
ldrDrago = tonemapDrago.process(hdrDebevec)
LDR_Drago = 3 * ldrDrago
plt.imshow(LDR_Drago[:,:,::-1])
plt.show()
# cv2.waitKey(0)
