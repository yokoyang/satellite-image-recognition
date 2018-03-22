# Python program to demonstrate erosion and
# dilation of images.
import cv2
import numpy as np

# Reading the input image
img = cv2.imread('../images/msk_prd_2.tif', 0)

# Taking a matrix of size 5 as the kernel
kernel = np.ones((8, 8), np.uint8)

# The first parameter is the original image,
# kernel is the matrix with which image is
# convolved and third parameter is the number
# of iterations, which will determine how much
# you want to erode/dilate a given image.
img_dilation = cv2.dilate(img, kernel, iterations=1)

img_erosion = cv2.erode(img_dilation, kernel, iterations=1)

cv2.imwrite('../images/Input.tif', img)
cv2.imwrite('../images/Erosion.tif', img_erosion)
cv2.imwrite('../images/Dilation.tif', img_dilation)

