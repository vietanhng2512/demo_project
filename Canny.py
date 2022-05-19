import cv2
import numpy as np

def scale_to_0_255(img):
    min_value = np.min(img)
    max_value = np.max(img)
    new_img = (img - min_value) / (max_value - min_value)
    new_img *= 255 # 0 - 1
    return new_img

def my_canny(img, min_value, max_value, sobel_size = 3, is_L2_gradient = False):
    """
    Try to implement Canny algorithm in OpenCV tutorial @ https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    """

    # Noise Reduction
    smooth_img = cv2.GaussianBlur(img, ksize = (5,5), sigmaX = 1, sigmaY = 1)

    # Finding Intensity Gradient of the Image
    Gx = cv2.Sobel(smooth_img, cv2.CV_64F, 1, 0, ksize = sobel_size)
    Gy = cv2.Sobel(smooth_img, cv2.CV_64F, 0, 1, ksize = sobel_size)

    if is_L2_gradient:
        edge_gradient = np.sqrt(Gx * Gx + Gy * Gy)
    else:
        edge_gradient = np.abs(Gx) + np.abs(Gy)

    angle = np.arctan2(Gy, Gx) * 180 / np.pi

    # round angle to 4 directions
    angle = np.abs(angle)
    angle[angle <= 22.5] = 0
    angle[angle >= 157.5] = 0
    angle[(angle > 22.5) * (angle < 67.5)] = 45
    angle[(angle >= 67.5) * (angle <= 112.5)] = 90
    angle[(angle > 112.5) * (angle <= 157.5)] = 135

    # Non-maximum Suppression
    keep_mask = np.zeros(smooth_img.shape, np.uint8)
    for y in range(1, edge_gradient.shape[0]-1):
        for x in range(1, edge_gradient.shape[1]-1):
            area_grad_intensity = edge_gradient[y-1:y-2, x-1:x+2] # 3x3 area
            area_angle = angle[y-1:y-2, x-1:x+2] # 3x3 area
            current_angle = area_angle[1, 1]
            current_angle_intensity = area_grad_intensity[1, 1]

            if current_angle == 0:
                if current_angle_intensity > max(area_grad_intensity[1,0], area_grad_intensity[1,2]):
                    keep_mask[y, x] = 255
                else:
                    edge_gradient[y, x] = 0
            elif current_angle == 45:
                if current_angle_intensity > max(area_grad_intensity[2,0],area_grad_intensity[0,2]):
                    keep_mask[y, x] = 255
                else:
                    edge_gradient[y,x] = 0
            elif current_angle == 90:
                if current_angle_intensity > max(area_grad_intensity[0,1], area_grad_intensity[2,1]):
                    keep_mask[y, x] = 255
                else:
                    edge_gradient[y, x] = 0
            elif current_angle == 135:
                if current_angle_intensity > max(area_grad_intensity[0,0], area_grad_intensity[2,2]):
                    keep_mask[y, x] = 255
                else:
                    edge_gradient[y, x] = 0

    # Hysteresis Thresholding
    canny_mask = np.zeros(smooth_img.shape, np.uint8)
    canny_mask[(keep_mask > 0) * (edge_gradient > min_value)] = 255

    return scale_to_0_255(canny_mask)

img = cv2.imread("girl.jpg",0)
my_canny = my_canny(img, min_value = 100, max_value = 200)
edges = cv2.Canny(img, 100, 200)
cv2.imshow("Anh", img)
cv2.imwrite("mycanny.jpg", img)
cv2.imshow("Anh lay canh", edges)
cv2.imwrite("edge.jpg", edges)

cv2.waitKey(0)






