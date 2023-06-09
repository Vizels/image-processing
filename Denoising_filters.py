from cv2 import BORDER_REPLICATE
import numpy as np
import cv2 as cv
import img_ctrl as ctrl

#convolition filter
def convolution_filter(img, mask_size):
    kernel = np.ones((mask_size,mask_size), np.float32)/mask_size**2
    out_img = cv.filter2D(img, -1, kernel)
    return out_img

def median_filter(img, mask_size):
    img = cv.copyMakeBorder(img, 1, 1, 1, 1, BORDER_REPLICATE) #add reflected border 1px (enlarge by 1)
    out_img = np.zeros((img.shape), np.int8) #creating empty output image
    height, width = out_img.shape[:2] 

    for i in range(1, height-1):
        for j in range(1, width-1):
            for color in range(3):
                out_img[i][j][color] = int(np.median(img[i-1:i+2, j-1:j+2, color])) # median
    
    return out_img

def bilateral_filter(img, d = 15, sigma_c = 75, sigma_s = 75):
    out_img = cv.bilateralFilter(img, d, sigma_c, sigma_s)
    return out_img

original = cv.imread('SIOC_labs/Leopard-1.jpg')
noised = cv.imread('SIOC_labs/Leopard-with-noise.jpg')


