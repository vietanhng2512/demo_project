import cv2
import numpy as np

img = cv2.imread("I04.jpg") # Load image
img_median = cv2.medianBlur(img, 7) # Add median filter to image
cv2.imshow("img", img_median) # Display with median filter
cv2.waitKey(0) # Wait for a key press to
cv2.destroyAllWindows()


