
'''
The following program is the implementation of Canny's Edge Detector algorithm in Python, which involves four major steps, namely, 
Gaussian smoothing, gradient operation, non-maxima suppression, and thresholding. The program accepts 
a grayscale image of size N * M (rows * columns) as input.
The output of the program at each of the four steps is as follows: 
1) an image result after Gaussian smoothing, 
(2) a normalized magnitude image, 
3) a normalized magnitude image after non-maxima suppression, and 
(4) binary edge maps for thresholds chosen at the 25th, 50th, and 75th percentiles 
(5) a histogram of the normalized magnitude image after non-maxima suppression.

Contributors:

Pragnavi RSD(pr2370)
Santosh Srinivas Ravichandran(sr6411)

'''

# Import necessary libraries
import numpy as np  # for numerical computations
from PIL import Image  # for image processing
import math  # for mathematical operations
import sys  # for system-level operations
import cv2  # for computer vision and image processing
import matplotlib.pyplot as plt #for plotting histograms and graphs
import os # using operating system dependent functionality like running system commands, extracting the file format extension from file path etc

# This function applies Gaussian smoothing on an input grayscale image using a pre-defined kernel.
# The function takes the image path as input and returns the smoothed image.

def gaussian_smoothing(img_path):
    
    # Load input image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Get image height and width
    img_h, img_w = img.shape

    # Define Gaussian kernel
    gaussian_mask = np.array([
        [1, 1, 2, 2, 2, 1, 1],
        [1, 2, 2, 4, 2, 2, 1],
        [2, 2, 4, 8, 4, 2, 2],
        [2, 4, 8, 16, 8, 4, 2],
        [2, 2, 4, 8, 4, 2, 2],
        [1, 2, 2, 4, 2, 2, 1],
        [1, 1, 2, 2, 2, 1, 1]
    ])
    
    # Get size of Gaussian kernel
    mask_size = gaussian_mask.shape[0]

    # Set buffer size for boundary pixels
    buffer = mask_size // 2

    # Initialize a 2D array to hold the smoothed image
    smooth_image = np.empty((img_h,img_w), dtype=np.float32)
    

    # Apply Gaussian smoothing using the pre-defined kernel
    for i in range(0+buffer,img_h-buffer,1):
        for j in range(0+buffer,img_w-buffer,1):
            # Calculate the weighted sum of the pixels in the kernel
            smooth_image[i][j] = \
                gaussian_mask[0][0]*img[i-buffer][j-buffer] \
                + gaussian_mask[0][1]*img[i-buffer][j-buffer+1] \
                + gaussian_mask[0][2]*img[i-buffer][j-buffer+2] \
                + gaussian_mask[0][3]*img[i-buffer][j-buffer+3] \
                + gaussian_mask[0][4]*img[i-buffer][j-buffer+4] \
                + gaussian_mask[0][5]*img[i-buffer][j-buffer+5] \
                + gaussian_mask[0][6]*img[i-buffer][j-buffer+6] \
                + gaussian_mask[1][0]*img[i-buffer+1][j-buffer] \
                + gaussian_mask[1][1]*img[i-buffer+1][j-buffer+1] \
                + gaussian_mask[1][2]*img[i-buffer+1][j-buffer+2] \
                + gaussian_mask[1][3]*img[i-buffer+1][j-buffer+3] \
                + gaussian_mask[1][4]*img[i-buffer+1][j-buffer+4] \
                + gaussian_mask[1][5]*img[i-buffer+1][j-buffer+5] \
                + gaussian_mask[1][6]*img[i-buffer+1][j-buffer+6] \
                + gaussian_mask[2][0]*img[i-buffer+2][j-buffer] \
                + gaussian_mask[2][1]*img[i-buffer+2][j-buffer+1] \
                + gaussian_mask[2][2]*img[i-buffer+2][j-buffer+2] \
                + gaussian_mask[2][3]*img[i-buffer+2][j-buffer+3] \
                + gaussian_mask[2][4]*img[i-buffer+2][j-buffer+4] \
                + gaussian_mask[2][5]*img[i-buffer+2][j-buffer+5] \
                + gaussian_mask[2][6]*img[i-buffer+2][j-buffer+6] \
                + gaussian_mask[3][0]*img[i-buffer+3][j-buffer] \
                + gaussian_mask[3][1]*img[i-buffer+3][j-buffer+1] \
                + gaussian_mask[3][2]*img[i-buffer+3][j-buffer+2] \
                + gaussian_mask[3][3]*img[i-buffer+3][j-buffer+3] \
                + gaussian_mask[3][4]*img[i-buffer+3][j-buffer+4] \
                + gaussian_mask[3][5]*img[i-buffer+3][j-buffer+5] \
                + gaussian_mask[3][6]*img[i-buffer+3][j-buffer+6] \
                + gaussian_mask[4][0]*img[i-buffer+4][j-buffer] \
                + gaussian_mask[4][1]*img[i-buffer+4][j-buffer+1] \
                + gaussian_mask[4][2]*img[i-buffer+4][j-buffer+2] \
                + gaussian_mask[4][3]*img[i-buffer+4][j-buffer+3] \
                + gaussian_mask[4][4]*img[i-buffer+4][j-buffer+4] \
                + gaussian_mask[4][5]*img[i-buffer+4][j-buffer+5] \
                + gaussian_mask[4][6]*img[i-buffer+4][j-buffer+6] \
                + gaussian_mask[5][0]*img[i-buffer+5][j-buffer] \
                + gaussian_mask[5][1]*img[i-buffer+5][j-buffer+1] \
                + gaussian_mask[5][2]*img[i-buffer+5][j-buffer+2] \
                + gaussian_mask[5][3]*img[i-buffer+5][j-buffer+3] \
                + gaussian_mask[5][4]*img[i-buffer+5][j-buffer+4] \
                + gaussian_mask[5][5]*img[i-buffer+5][j-buffer+5] \
                + gaussian_mask[5][6]*img[i-buffer+5][j-buffer+6] \
                + gaussian_mask[6][0]*img[i-buffer+6][j-buffer] \
                + gaussian_mask[6][1]*img[i-buffer+6][j-buffer+1] \
                + gaussian_mask[6][2]*img[i-buffer+6][j-buffer+2] \
                + gaussian_mask[6][3]*img[i-buffer+6][j-buffer+3] \
                + gaussian_mask[6][4]*img[i-buffer+6][j-buffer+4] \
                + gaussian_mask[6][5]*img[i-buffer+6][j-buffer+5] \
                + gaussian_mask[6][6]*img[i-buffer+6][j-buffer+6] \

    # Normalize the smoothed image by dividing with the sum of Gaussian kernel elements
    smooth_image = smooth_image/np.sum(gaussian_mask)
    
    # Show the smoothed image
    im = Image.fromarray(smooth_image).convert('L')
    im.show()

    # Save the smoothed image in the format 'Gaussian_Smoothened_imagename.bmp'
    im.save('1_Gaussian_Smoothened_'+img_file)
    
    return smooth_image


# This function computes the Gradient operation on the input gaussian smoothed image using predefined masks from the Robinson compass mask for edge detection.
# The function takes the gaussian smoothed image array as input and returns the normalized edge magnitude array and the gradient angle array. 
   
def gradient_operation(gaussian_smooth_image):

    img = gaussian_smooth_image
    img_h, img_w = img.shape

    #Defining the gradient masks - g0 for 0 degree, g1 for 45 degree, g2 for 90 degree & g3 for 135 degree. 
    g0 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    g1 = np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
    g2 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    g3 = np.array([[2,1,0],[1,0,-1],[0,-1,-2]])

    # Get size of Gradient masks
    mask_size = g0.shape[0]

    # Set buffer size for boundary pixels
    buffer = mask_size // 2

    # Initialize 2D array to hold each output image of gradient operation on the input image using each masks

    h0 = np.zeros((img_h,img_w), dtype=np.float32)
    h1 = np.zeros((img_h,img_w), dtype=np.float32)
    h2 = np.zeros((img_h,img_w), dtype=np.float32)
    h3 = np.zeros((img_h,img_w), dtype=np.float32)

    # Apply Gradient operation using the pre-defined gradient masks

    for i in range(0+buffer,img_h-buffer,1):
            for j in range(0+buffer,img_w-buffer,1):

                # Calculate the weighted sum of the pixels in g0 mask
                h0[i][j] = \
                    g0[0][0]*img[i-buffer][j-buffer] \
                    + g0[0][1]*img[i-buffer][j-buffer+1] \
                    + g0[0][2]*img[i-buffer][j-buffer+2] \
                    + g0[1][0]*img[i-buffer+1][j-buffer] \
                    + g0[1][1]*img[i-buffer+1][j-buffer+1] \
                    + g0[1][2]*img[i-buffer+1][j-buffer+2] \
                    + g0[2][0]*img[i-buffer+2][j-buffer] \
                    + g0[2][1]*img[i-buffer+2][j-buffer+1] \
                    + g0[2][2]*img[i-buffer+2][j-buffer+2] \

                # Calculate the weighted sum of the pixels in g1 mask
                h1[i][j] = \
                    g1[0][0]*img[i-buffer][j-buffer] \
                    + g1[0][1]*img[i-buffer][j-buffer+1] \
                    + g1[0][2]*img[i-buffer][j-buffer+2] \
                    + g1[1][0]*img[i-buffer+1][j-buffer] \
                    + g1[1][1]*img[i-buffer+1][j-buffer+1] \
                    + g1[1][2]*img[i-buffer+1][j-buffer+2] \
                    + g1[2][0]*img[i-buffer+2][j-buffer] \
                    + g1[2][1]*img[i-buffer+2][j-buffer+1] \
                    + g1[2][2]*img[i-buffer+2][j-buffer+2] \
                               
                # Calculate the weighted sum of the pixels in g2 mask
                h2[i][j] = \
                    g2[0][0]*img[i-buffer][j-buffer] \
                    + g2[0][1]*img[i-buffer][j-buffer+1] \
                    + g2[0][2]*img[i-buffer][j-buffer+2] \
                    + g2[1][0]*img[i-buffer+1][j-buffer] \
                    + g2[1][1]*img[i-buffer+1][j-buffer+1] \
                    + g2[1][2]*img[i-buffer+1][j-buffer+2] \
                    + g2[2][0]*img[i-buffer+2][j-buffer] \
                    + g2[2][1]*img[i-buffer+2][j-buffer+1] \
                    + g2[2][2]*img[i-buffer+2][j-buffer+2] \

                # Calculate the weighted sum of the pixels in g3 mask
                h3[i][j] = \
                    g3[0][0]*img[i-buffer][j-buffer] \
                    + g3[0][1]*img[i-buffer][j-buffer+1] \
                    + g3[0][2]*img[i-buffer][j-buffer+2] \
                    + g3[1][0]*img[i-buffer+1][j-buffer] \
                    + g3[1][1]*img[i-buffer+1][j-buffer+1] \
                    + g3[1][2]*img[i-buffer+1][j-buffer+2] \
                    + g3[2][0]*img[i-buffer+2][j-buffer] \
                    + g3[2][1]*img[i-buffer+2][j-buffer+1] \
                    + g3[2][2]*img[i-buffer+2][j-buffer+2] \


    # Take the absolute value of each response after gradient operation
    h0 = abs(h0)
    h1 = abs(h1)
    h2 = abs(h2)
    h3 = abs(h3)

    # Find the maximum value among the four gradient directions at each location
    max_arr = np.maximum(np.maximum(np.maximum(h0, h1), h2), h3)
    
    # Find the indices where the maximum value occurs for each gradient direction (0,45,90 or 135 degree)
    h0_indices = np.where(h0 == max_arr)
    h1_indices = np.where(h1 == max_arr)
    h2_indices = np.where(h2 == max_arr)
    h3_indices = np.where(h3 == max_arr)
    
    # Create an array to store the gradient angle for each pixel
    gradient_angle = np.zeros((img_h,img_w), dtype=np.float32)

    # Merge the indices for each gradient direction into a list
    merged_h0 = [list(row) for row in zip(*h0_indices)]
    merged_h1 = [list(row) for row in zip(*h1_indices)]
    merged_h2 = [list(row) for row in zip(*h2_indices)]
    merged_h3 = [list(row) for row in zip(*h3_indices)]
    
    # Assign the gradient angle for each pixel based on the maximum gradient direction
    for ind in merged_h0:
        i,j = ind
        gradient_angle[i][j] = 0
    
    for ind in merged_h1:
        i,j = ind
        gradient_angle[i][j] = 45

    for ind in merged_h2:
        i,j = ind
        gradient_angle[i][j] = 90

    for ind in merged_h3:
        i,j = ind
        gradient_angle[i][j] = 135
        
    # Normalize the maximum gradient values by dividing by 4
    normalized_array = max_arr/4

    # Set the edge pixels to 0 where part of the mask goes outside of the image border or lies in the undefined region of the image after Gaussian filtering
    normalized_array[:4, :] = 0
    normalized_array[:, :4] = 0
    normalized_array[-4:, :] = 0
    normalized_array[:, -4:] = 0

    gradient_angle[:4, :] = 0
    gradient_angle[:, :4] = 0
    gradient_angle[-4:, :] = 0
    gradient_angle[:, -4:] = 0

    # Show the normalized gradient magnitude image
    im = Image.fromarray(normalized_array).convert('L')
    im.show()

    # Save the normalized gradient magnitude  image in the format '2_Gradient_Magnitude_imagename.bmp'
    im.save('2_Gradient_Magnitude_'+img_file)

    return normalized_array, gradient_angle

# This function performs non-maximum suppression on a gradient magnitude array, based on the gradient angle array. 
# The resulting non-maximum suppression (NMS) array and the list of gradient magnitude values after nms excluding 0 are returned.

def non_maxima_suppression(gradient_magnitude_array, gradient_angle_array):

    img = gradient_magnitude_array
    img_h, img_w = img.shape

    # initialize sector matrix
    sector = np.zeros((img_h, img_w))

    # quantize angle into 4 sectors
    for i in range(0,img_h,1):
        for j in range(0,img_w,1):
            if 0 <= gradient_angle_array[i,j] < 22.5 :
                sector[i,j] = 0
            elif 22.5 <= gradient_angle_array[i,j] < 67.5 :
                sector[i,j] = 1
            elif 67.5 <= gradient_angle_array[i,j] < 112.5 :
                sector[i,j] = 2
            elif 112.5 <= gradient_angle_array[i,j] < 157.5 :
                sector[i,j] = 3
            elif 157.5 <= gradient_angle_array[i,j] < 202.5 :
                sector[i,j] = 0
            elif 202.5 <= gradient_angle_array[i,j] < 247.5 :
                sector[i,j] = 1
            elif 247.5 <= gradient_angle_array[i,j] < 292.5 :
                sector[i,j] = 2
            elif 292.5 <= gradient_angle_array[i,j] < 337.5 :
                sector[i,j] = 3
            elif 337.5 <= gradient_angle_array[i,j] <= 360:
                sector[i,j] = 0
            else:
                sector[i,j] = -1

    # define buffer
    buffer = 4+1

    # initialize nms and magnitude_arr (for percentile calculation later)
    nms = np.zeros((img_h, img_w))
    magnitude_arr = []

    # performing non maxima suppression by comparing gradient magnitudes according to quantized sectors
    for i in range(0+buffer,img_h-buffer,1):
        for j in range(0+buffer,img_w-buffer,1):
            if sector[i,j] == 2:
                # compare to upper and lower magnitudes
                if ( gradient_magnitude_array[i][j] > gradient_magnitude_array[i-1][j] ) \
                    and ( gradient_magnitude_array[i][j] > gradient_magnitude_array[i+1][j] ) :
                    nms[i,j] = gradient_magnitude_array[i][j]
                    magnitude_arr.append(gradient_magnitude_array[i][j])
                else :
                    nms[i,j] = 0
            elif sector[i,j] == 3:
                # compare to upper left and lower right mag
                if ( gradient_magnitude_array[i][j] > gradient_magnitude_array[i-1][j-1] ) \
                    and ( gradient_magnitude_array[i][j] > gradient_magnitude_array[i+1][j+1] ) :
                    nms[i,j] = gradient_magnitude_array[i][j]
                    magnitude_arr.append(gradient_magnitude_array[i][j])
                else :
                    nms[i,j] = 0
            elif sector[i,j] == 0:
                # compare to right and left mag
                if ( gradient_magnitude_array[i][j] > gradient_magnitude_array[i][j-1] ) \
                    and ( gradient_magnitude_array[i][j] > gradient_magnitude_array[i][j+1] ) :
                    nms[i,j] = gradient_magnitude_array[i][j]
                    magnitude_arr.append(gradient_magnitude_array[i][j])
                else :
                    nms[i,j] = 0
            elif sector[i,j] == 1 :
                # compare to upper right and lower left mag
                if ( gradient_magnitude_array[i][j] > gradient_magnitude_array[i-1][j+1] ) \
                    and ( gradient_magnitude_array[i][j] > gradient_magnitude_array[i+1][j-1] ) :
                    nms[i,j] = gradient_magnitude_array[i][j]
                    magnitude_arr.append(gradient_magnitude_array[i][j])
                else :
                    nms[i,j] = 0
            elif sector[i,j] == -1:
                # suppress to zero
                nms[i,j] = 0
    # show nms
    im = Image.fromarray(nms).convert("L")
    im.show()

    # save nms (4)
    im.save('3_NMS_'+img_file)

    return nms, magnitude_arr

#This function applies thresholding to the non-maximum suppression (NMS) array and generates three binary edge maps 
#with edge pixels set on three different thresholds from 25th, 50th and 75th percentile of gradient magnitude array after nms excluding 0.  

def thresholding(nms_array, magnitude_arr):

    img_h, img_w = nms_array.shape

    # calculate thresholds from percentiles
    T1 = np.percentile(magnitude_arr,25)
    T2 = np.percentile(magnitude_arr,50)
    T3 = np.percentile(magnitude_arr,75)

    # initialize final threshold images
    final_threshold_img_t1 = np.zeros((img_h, img_w))
    final_threshold_img_t2 = np.zeros((img_h, img_w))
    final_threshold_img_t3 = np.zeros((img_h, img_w))

    # apply threshold and generate binary edge map
    for i in range(0,img_h,1):
        for j in range(0,img_w,1):
            if nms_array[i][j] >=T1:
                final_threshold_img_t1[i][j] = 255
            if nms_array[i][j] >=T2:
                final_threshold_img_t2[i][j] = 255
            if nms_array[i][j] >=T3:
                final_threshold_img_t3[i][j] = 255

    # show final image T1
    im = Image.fromarray(final_threshold_img_t1).convert('1')
    im.show()

    # save final image T1 (5)
    im.save('4_Binary_Edge_Map_T1_'+img_file)

    # show final image T2
    im = Image.fromarray(final_threshold_img_t2).convert('1')
    im.show()

    # save final image T2 (5)
    im.save('5_Binary_Edge_Map_T2_'+img_file)

    # show final image T3
    im = Image.fromarray(final_threshold_img_t3).convert('1')
    im.show()

    # save final image T3 (5)
    im.save('6_Binary_Edge_Map_T3_'+img_file)

    return final_threshold_img_t1, final_threshold_img_t2, final_threshold_img_t3

#This function takes in nms array as input and generates a histogram of the gradient magnitudes after nms.

def nms_histogram(nms_array):

    max_pixels = 512*512

    gray_image = np.random.choice(nms_array.flatten(), size=max_pixels)

    # add axis labels and title
    plt.xlabel('Gradient Magnitudes after NMS')
    plt.ylabel('No. of Pixels')
    plt.title('Histogram of Gradient values after NMS')

    # create histogram plot with bins as 256
    plt.hist(nms_array.flatten(), bins=256, range=(0, 255), color='gray')

    #save histogram image (8)
    plt.savefig('7_nms_histogram_'+os.path.splitext(img_file)[0] + '.png')

    # display the plot
    plt.show()


# ensure proper arguments given
if (len(sys.argv)) < 2:
    print("Command failure. Usage: $ python3 canny.py [image_file_name].bmp")
    exit()

# show input image from parameter passed
img_file = sys.argv[1]

try:
    img = Image.open(img_file).convert('L')
    img.show()
except:
    print("Error loading image")
    sys.exit(1)
    
# convert list to numpy array
input_img = np.array(img)

# compute input dimentions
height, width = input_img.shape

# canny edge detection steps

# 1. Gaussian smoothing
smooth_image = gaussian_smoothing(img_file)
# 2. Gradient Operation
gradient_magnitude, gradient_angle  = gradient_operation(smooth_image)
# 3. NMS
nms, magnitude_arr = non_maxima_suppression(gradient_magnitude, gradient_angle)
# 4. Thresholding
final_threshold_img_t1, final_threshold_img_t2, final_threshold_img_t3 = thresholding(nms, magnitude_arr)
#5 . Histogram
nms_histogram(nms)
