import numpy as np
import cv2 as cv


def bayer_mosaic(img):
    (height, width) = img.shape[:2]
    bayer = np.zeros((height, width, 3), np.uint8)

    #GREEN pxls TOP LEFT
    for i in range(0, height, 2): 
        for j in range (0, width, 2):
            color_level = img[i,j,1]
            bayer[i, j] = [0, color_level, 0]

    #RED TOP RIGHT
    for i in range(0, height, 2):
        for j in range (1, width, 2):
            color_level = img[i,j,2]
            bayer[i, j] = [0, 0, color_level]

    #BLUE BOT LEFT        
    for i in range(1, height, 2):
        for j in range (0, width, 2):
            color_level = img[i,j,0]
            bayer[i, j] = [color_level, 0, 0]

    #GREEN BOT RIGHT
    for i in range(1, height, 2):
        for j in range (1, width, 2):
            color_level = img[i,j,1]
            bayer[i, j] = [0, color_level, 0]

    return bayer


# [G][B][R][G][R][B]    [0][1][2][3][4][5]
# [R][G][G][B][G][G]    [6][7][8][9][10][11]
# [B][G][G][R][G][G]    [12][13][14][15][16][17]
# [G][R][B][G][B][R]    [18][19][20][21][22][23]
# [B][G][G][R][G][G]    [24][25][26][27][28][29]
# [R][G][G][B][G][G]    [30][31][32][33][34][35]
def fuji_x_trans_mosaic(img):
    (height, width) = img.shape[:2]
    add_bot = 6-(height%6)
    add_right = 6-(width%6)
    img = cv.copyMakeBorder(img, 0, add_bot, 0, add_right, cv.BORDER_CONSTANT)
    (height, width) = img.shape[:2]

    fuji = np.zeros((height, width, 3), np.uint8)

    #RED 6,15, 27,30
    for i in range(1, height, 6):
        for j in range (0, width, 6):
            for k in range (2):
                color_level = img[i+4*k,j,2]
                fuji[i+4*k, j] = [0, 0, color_level]

                color_level = img[i+1+k*2,j+3,2]
                fuji[i+1+k*2, j+3] = [0, 0, color_level]


    #RED 2,4, 19,23
    for i in range(0, height, 6):
        for j in range (2, width, 6):
            for k in range (2):
                color_level = img[i,j+2*k,2]
                fuji[i, j+2*k] = [0, 0, color_level]

                color_level = img[i+3,j-1+4*k,2]
                fuji[i+3, j-1+4*k] = [0, 0, color_level]

    #BLUE 1,5, 20,22
    for i in range(0, height, 6):
        for j in range(1, width, 6):
            for k in range(2):
                color_level = img[i,j+4*k,0]
                fuji[i, j+4*k] = [color_level, 0, 0]

                color_level = img[i+3,j+1+k*2,0]
                fuji[i+3, j+1+k*2] = [color_level, 0, 0]

    #BLUE 9,12, 24,33
    for i in range(2, height, 6):
        for j in range(0, width, 6):
            for k in range(2):
                color_level = img[i+k*2,j,0]
                fuji[i+k*2, j] = [color_level, 0, 0]

                color_level = img[i-1+k*4,j+3,0]
                fuji[i-1+k*4, j+3] = [color_level, 0, 0]


    #GREEN 0, 2, 18, 21
    for i in range(0, height, 3):
        for j in range(0, width, 3):
            color_level = img[i,j,1]
            fuji[i, j] = [0, color_level, 0]
    
    #GREEN 7,8, 10,11, 13,14 etc.
    for i in range(1, height,3):
        for j in range(1, width,3):
            for k in range(2):
                for h in range(2):
                    color_level = img[i+k,j+h,1]
                    fuji[i+k, j+h] = [0, color_level, 0]
                    


    fuji = fuji[0:height-add_bot, 0:width-add_right]
    

    return fuji



def fuji_demosaic(img):
    (height, width) = img.shape[:2]
    demosaic = np.empty((height, width, 3), np.uint8)

    B, G, R = cv.split(img)

    kernelG = np.ones((3,3), np.float32)/5
    kernelBR = np.ones((3,6), np.float32)/4
    demosaic[:,:, 1] = cv.filter2D(G, -1, kernelG) #G
    demosaic[:,:, 0] = cv.filter2D(B, -1, kernelBR) #B
    demosaic[:,:, 2] = cv.filter2D(R, -1, kernelBR) #R

    return demosaic

def bayer_demosaic(img):
    (height, width) = img.shape[:2]
    demosaic = np.zeros((height, width, 3), np.uint8)

    B, G, R = cv.split(img)

    kernelG = np.ones((2,2), np.float32)/2
    kernelBR = np.ones((2,2), np.float32)
    demosaic[:,:, 1] = cv.filter2D(G, -1, kernelG) #G
    demosaic[:,:, 0] = cv.filter2D(B, -1, kernelBR) #B
    demosaic[:,:, 2] = cv.filter2D(R, -1, kernelBR) #R

    return demosaic



img = cv.imread('f:\Projects\Python\SIOC_labs\demosaicking.bmp')

#mosaic_bayer = bayer_mosaic(img)

mosaic_fuji = fuji_x_trans_mosaic(img)
demosaic_bayer = bayer_demosaic(bayer_mosaic(img))
demosaic_fuji = fuji_demosaic(mosaic_fuji)
cv.imwrite('fuji_mosaiced.bmp', mosaic_fuji)
cv.imwrite('bayer_demosaiced.png', demosaic_bayer)
cv.imwrite('fuji_demosaiced.png', demosaic_fuji)
