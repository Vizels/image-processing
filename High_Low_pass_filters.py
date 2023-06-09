import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os
import pywt
import pywt.data
import cv2

# Load image
img = cv2.imread("Lenna.png")
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fourier = np.fft.fft2(grey)
dft_shift = np.fft.fftshift(fourier)



def highPassFiltering(img, size):

    h, w = img.shape[0:2]
    h1, w1 = int(h / 2), int(w / 2)  
    img[h1-size:h1+size, w1-size:w1+size] = 0

    return img

def lowPassFiltering(img, size):

    h, w = img.shape[0:2]
    h1, w1 = int(h / 2), int(w / 2)  
    img[0:h, 0:w1 - int(size / 2)] = 0
    img[0:h, w1 + int(size / 2):w] = 0
    img[0:h1 - int(size / 2), 0:w] = 0
    img[h1 + int(size / 2):h, 0:w] = 0

    return img


#lowpassed = lowPassFiltering(dft_shift, 4)

filtered = highPassFiltering(dft_shift, 100)


res = np.log(np.abs(dft_shift))

idft_shift = np.fft.ifftshift(filtered)
ifimg = np.fft.ifft2(idft_shift)
ifimg = np.abs(ifimg)

# dft_shift_lowpassed = lowPassFiltering(dft_shift)

# res =np.log(np.abs(dft_shift_lowpassed))

# idft_shift = np.fft.ifftshift(dft_shift_lowpassed)

cv2.imshow("gray", ifimg)
#cv2.imwrite("highpass_filter_100.png", ifimg)
cv2.waitKey(0)