import numpy as np
from PIL import Image

############### ---------- Basic Image Processing ------ ##############
def play():
    print("hi")

### TODO 1: Read an Image and convert it into a floating point array with values between 0 and 1. You can assume a color image
def imread(filename):
    image = Image.open(filename)
    image_data = np.asarray(image).astype(float)
    image_data /= 256
    return image_data

### TODO 2: Convolve an image (m x n x 3 or m x n) with a filter(l x k). Perform "same" filtering. Apply the filter to each channel if there are more than 1 channels
def convolve(img, filt):
    if len(img.shape) == 2:
        return convolve2D(img, filt)
    else:
        result = np.zeros(img.shape)
        for i in range(3):
            img2d = img[:, :, i]
            result[:, :, i] = convolve2D(img2d, filt)
        return result


def convolve2D(img, filt):
    m = len(img)
    n = len(img[0])
    l = len(filt)
    k = len(filt[0])
    result = np.zeros(img.shape)
    for i in range(m):
        for j in range(n):
            num = 0
            for a in range(l):
                for b in range(k):
                    # filt posi: a, b
                    # img posi: i-(a-(l-1)/2), j-(b-(k-1)/2)
                    filt_x = a
                    filt_y = b
                    img_x = i-(a-(l-1)//2)
                    img_y = j-(b-(k-1)//2)
                    
                    if (img_x < 0) or (img_x >= m) or (img_y < 0) or (img_y >=n):
                        num += 0
                    else:
                        num += img[img_x][img_y] * filt[filt_x][filt_y]
                        #print(img[img_x][img_y])
            result[i][j] = num
    return result

### TODO 3: Create a gaussian filter of size k x k and with standard deviation sigma
def gaussian_filter(k, sigma):
    result = np.zeros((k, k))
    n = (k-1) // 2
    for i in range(-n, n, 1):
        for j in range(-n, n, 1):
            result[i+n][j+n] = gaussian_val(i, j, sigma)
    all_sum = np.sum(result)
    for i in range(-n, n, 1):
        for j in range(-n, n, 1):
            result[i+n][j+n] /= all_sum
    return result

def gaussian_val(x, y, sigma):
    temp1 = 1/ (2 * np.pi * (sigma**2))
    temp2 = np.exp(-(x**2 + y**2) / (2 * (sigma**2)))
    return temp1 * temp2


### TODO 4: Compute the image gradient. 
### First convert the image to grayscale by using the formula:
### Intensity = Y = 0.2125 R + 0.7154 G + 0.0721 B
### Then convolve with a 5x5 Gaussian with standard deviation 1 to smooth out noise. 
### Convolve with [0.5, 0, -0.5] to get the X derivative on each channel
### convolve with [[0.5],[0],[-0.5]] to get the Y derivative on each channel
### Return the gradient magnitude and the gradient orientation (use arctan2)
def gradient(img):
    pass

##########----------------Line detection----------------

### TODO 5: Write a function to check the distance of a set of pixels from a line parametrized by theta and c. The equation of the line is:
### x cos(theta) + y sin(theta) + c = 0
### The input x and y are numpy arrays of the same shape, representing the x and y coordinates of each pixel
### Return a boolean array that indicates True for pixels whose distance is less than the threshold
def check_distance_from_line(x, y, theta, c, thresh):
    pass

### TODO 6: Write a function to draw a set of lines on the image. 
### The `img` input is a numpy array of shape (m x n x 3).
### The `lines` input is a list of (theta, c) pairs. 
### Mark the pixels that are less than `thresh` units away from the line with red color,
### and return a copy of the `img` with lines.
def draw_lines(img, lines, thresh):
    pass

### TODO 7: Do Hough voting. You get as input the gradient magnitude (m x n) and the gradient orientation (m x n), 
### as well as a set of possible theta values and a set of possible c values. 
### If there are T entries in thetas and C entries in cs, the output should be a T x C array. 
### Each pixel in the image should vote for (theta, c) if:
### (a) Its gradient magnitude is greater than thresh1, **and** 
### (b) Its distance from the (theta, c) line is less than thresh2, **and**
### (c) The difference between theta and the pixel's gradient orientation is less than thresh3
def hough_voting(gradmag, gradori, thetas, cs, thresh1, thresh2, thresh3):
    pass   

### TODO 8: Find local maxima in the array of votes. A (theta, c) pair counts as a local maxima if: 
### (a) Its votes are greater than thresh, **and** 
### (b) Its value is the maximum in a nbhd x nbhd beighborhood in the votes array.
### The input `nbhd` is an odd integer, and the nbhd x nbhd neighborhood is defined with the 
### coordinate of the potential local maxima placing at the center.
### Return a list of (theta, c) pairs.
def localmax(votes, thetas, cs, thresh, nbhd):
    pass
  
# Final product: Identify lines using the Hough transform    
def do_hough_lines(filename):

    # Read image in
    img = imread(filename)

    # Compute gradient
    gradmag, gradori = gradient(img)

    # Possible theta and c values
    thetas = np.arange(-np.pi-np.pi/40, np.pi+np.pi/40, np.pi/40)
    imgdiagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    cs = np.arange(-imgdiagonal, imgdiagonal, 0.5)

    # Perform Hough voting
    votes = hough_voting(gradmag, gradori, thetas, cs, 0.1, 0.5, np.pi/40)

    # Identify local maxima to get lines
    lines = localmax(votes, thetas, cs, 20, 11)

    # Visualize: draw lines on image
    result_img = draw_lines(img, lines, 0.5)

    # Return visualization and lines
    return result_img, lines
   
    
