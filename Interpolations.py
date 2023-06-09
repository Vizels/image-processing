import numpy as np
import cv2 as cv #BGR
import math
import time

original = cv.imread('f:\Projects\Python\SIOC_labs\macaw.bmp')

def reduce_N_times(image, N:np.uint8, prev_blur = False):

    #get shape
    height, width = image.shape[:2]

    #get smaller shapes
    small_height, small_width = int(height/N), int(width/N)

    #create empty image
    img_small = np.zeros((small_height, small_width, 3), dtype=np.uint8)


    # blur original for smoother small image
    if prev_blur == True:
        kernel = np.ones((N, N))/(N*N)
        image = cv.filter2D(image, -1, kernel)

    #reducin algorithm
    for i in range(0, small_height):
        for j in range(0, small_width):
            img_small[i][j] = image[i*N][j*N]
    return img_small

def enlarged_Nx(image, N:np.uint8, interpolation = 'no'):
    #get shape
    height, width = image.shape[:2]

    #get smaller shapes
    large_height, large_width = height*N, width*N

    #create empty pic
    img_big = np.zeros((large_height, large_width, 3), dtype=np.uint8)

    #enlargin algorithm
    for i in range(0, height):
        for j in range(0, width):
            img_big[i*N][j*N] = image[i][j]

    #NO INTERPOLATION
    if interpolation == 'no':
        interpolated = img_big

    #NEAREST NEIGHBOUR
    if interpolation == 'nearest':
        kernel_simple = np.ones((N,N), dtype=np.float32)
        interpolated = cv.filter2D(img_big, -1, kernel_simple)

    #BILINEAR
    if interpolation == 'bilinear' and N == 2:
        #bilinear kernel
        #! only for 2x
        kernel_bilinear = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32)/4
        interpolated = cv.filter2D(img_big, -1, kernel_bilinear)
    elif interpolation == 'bilinear' and N != 2:
        interpolated = cv.resize(image, (height*N, width*N), interpolation=cv.INTER_LINEAR)

    #BICUBIC
    if interpolation == 'bicubic':
        interpolated = cv.resize(image, (height*N, width*N), interpolation=cv.INTER_CUBIC)
        

    return interpolated

def rotated_Xdeg(img, degree):
    #degrees to rads
    rads = math.radians(degree)


    height, width = img.shape[:2]
    rot_img = np.zeros((height, width, 3), dtype=np.uint8)

    #center point
    midx, midy = (width//2, height//2)

    #rotation algorithm
    for i in range(height):
        for j in range(width):
            #rotation matrix
            x = (i-midx)*math.cos(rads) + (j-midy)*math.sin(rads)
            y = -(i-midx)*math.sin(rads) + (j-midy)*math.cos(rads)
            
            x=round(x)+midx 
            y=round(y)+midy 

            if (x>=0 and y>=0 and x<img.shape[0] and y<img.shape[1]):
                rot_img[i,j,:] = img[x,y,:]

    return rot_img 

small_2x = reduce_N_times(original, 2)
large_2x = enlarged_Nx(small_2x, 2, interpolation='bicubic')
rotated_45deg = rotated_Xdeg(original, 45)


def show():
    cv.imshow("Original", original)
    cv.imshow("2x reduced",small_2x)
    cv.imshow("2x enlarged", large_2x)
    cv.imshow("Rotated 45 degrees", rotated_45deg)
    

    cv.waitKey(0)

small = cv.imread('2x_reduced.bmp')

def save():
    #cv.imwrite("2x_reduced.bmp", reduce_N_times(original, 2))

    start_time = time.time()
    cv.imwrite("small_to_orig_no_interpolation.png", enlarged_Nx(small, 2)) #no interpolation (default)
    print("No interpolation enlrage image:", (time.time()-start_time)*1000, 'ms')
    start_time = time.time()

    start_time = time.time()
    cv.imwrite("small_to_orig_NN.png", enlarged_Nx(small, 2, 'nearest') )
    print("Nearst neighbour interpolation time enlrage image:", (time.time()-start_time)*1000, 'ms')

    start_time = time.time()
    cv.imwrite("small_to_orig_BL.png", enlarged_Nx(small, 2, 'bilinear'))
    print("Bilinear interpolation time enlrage image:", (time.time()-start_time)*1000, 'ms')

    start_time = time.time()
    cv.imwrite("small_to_orig_BC.png", enlarged_Nx(small, 2, 'bicubic')) 
    print("Bicubic interpolation time enlrage image:", (time.time()-start_time)*1000, 'ms')

    #cv.imwrite("enlarged_to_original.bmp", large_2x)
    #cv.imwrite("Rotated_45_deg.bmp", rotated_45deg)
    #cv.imwrite("Rotated_30_deg.bmp", rotated_Xdeg(original, 30))

cv.imwrite("rotated_45deg.png",rotated_45deg)