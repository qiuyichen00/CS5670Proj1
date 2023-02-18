import numpy as np
from PIL import Image

############### ---------- Basic Image Processing ------ ##############
def play():
    print("hi")

### TODO 1: Read an Image and convert it into a floating point array with values between 0 and 1. You can assume a color image
def imread(filename):
    image = Image.open(filename)
    image.convert('RGB')
    image_data = np.asarray(image).astype(float)
    image_data /= 255
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
    m = img.shape[0]
    n = img.shape[1]
    l = filt.shape[0]
    k = filt.shape[1]
    result = np.zeros(img.shape)
    # for each pixel
    for i in range(m):
        for j in range(n):
            newval = 0
            # for each filter position a, b
            for a in range(l): 
                for b in range(k):
                    img_x = i-(a-(l-1)//2)
                    img_y = j-(b-(k-1)//2)
                    if (img_x < 0) or (img_x >= m) or (img_y < 0) or (img_y >=n):
                        newval += 0
                    else:
                        newval += img[img_x][img_y] * filt[a][b]
            result[i][j] = newval
    return result


### TODO 3: Create a gaussian filter of size k x k and with standard deviation sigma
def gaussian_filter(k, sigma):
    result = np.zeros((k, k))
    n = (k-1) // 2
    all_sum = 0
    for i in range(-n, n + 1, 1):
        for j in range(-n, n + 1, 1):
            val = gaussian_val(i, j, sigma)
            result[i+n][j+n] = val
            all_sum += val

    return result / all_sum

def gaussian_val(x, y, sigma):
    temp1 = 1 / (2 * np.pi * (sigma**2))
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
    img_gs = grey_scale(img)
    img_gsfilt = convolve(img_gs, gaussian_filter(5, 1))
    img_dx = convolve(img_gsfilt, np.array([[0.5, 0, -0.5]]))
    img_dy = convolve(img_gsfilt, np.array([[0.5], [0], [-0.5]]))
    return np.sqrt(np.square(img_dx) + np.square(img_dy)), np.arctan2(img_dy, img_dx)

def grey_scale(img):
    return img[:, :, 0] * 0.2125 + img[:, :, 1] * 0.7154 + img[:, :, 2] * 0.0721

##########----------------Line detection----------------

### TODO 5: Write a function to check the distance of a set of pixels from a line parametrized by theta and c. The equation of the line is:
### x cos(theta) + y sin(theta) + c = 0
### The input x and y are numpy arrays of the same shape, representing the x and y coordinates of each pixel
### Return a boolean array that indicates True for pixels whose distance is less than the threshold
def check_distance_from_line(x, y, theta, c, thresh):
    return np.abs(x * np.cos(theta) + y * np.sin(theta) + c) < thresh

### TODO 6: Write a function to draw a set of lines on the image. 
### The `img` input is a numpy array of shape (m x n x 3).
### The `lines` input is a list of (theta, c) pairs. 
### Mark the pixels that are less than `thresh` units away from the line with red color,
### and return a copy of the `img` with lines.
def draw_lines(img, lines, thresh):
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            for (theta,c) in lines:
                if check_distance_from_line(x, y, theta, c, thresh):
                        img[y][x] = [1, 0, 0]
    return img


### TODO 7: Do Hough voting. You get as input the gradient magnitude (m x n) and the gradient orientation (m x n), 
### as well as a set of possible theta values and a set of possible c values. 
### If there are T entries in thetas and C entries in cs, the output should be a T x C array. 
### Each pixel in the image should vote for (theta, c) if:
### (a) Its gradient magnitude is greater than thresh1, **and** 
### (b) Its distance from the (theta, c) line is less than thresh2, **and**
### (c) The difference between theta and the pixel's gradient orientation is less than thresh3
def hough_voting(gradmag, gradori, thetas, cs, thresh1, thresh2, thresh3):

    result = np.zeros([thetas.shape[0], cs.shape[0]])
    x = np.zeros_like(gradmag)
    y = np.zeros_like(gradmag)
    for i in range(gradmag.shape[0]):
        for j in range(gradmag.shape[1]):
            x[i][j] = j
            y[i][j] = i
    
    cond1 = gradmag > thresh1
    for t in range(thetas.shape[0]):
        cond3 = (np.abs(gradori - thetas[t]) < thresh3)
        for c in range(cs.shape[0]):
            cond2 = check_distance_from_line(x, y, thetas[t], cs[c], thresh2)
            vote_map = np.logical_and(np.logical_and(cond1, cond2),cond3)
            votes = np.sum(vote_map)
            result[t][c] += votes

    return result


### TODO 8: Find local maxima in the array of votes. A (theta, c) pair counts as a local maxima if: 
### (a) Its votes are greater than thresh, **and** 
### (b) Its value is the maximum in a nbhd x nbhd beighborhood in the votes array.
### The input `nbhd` is an odd integer, and the nbhd x nbhd neighborhood is defined with the 
### coordinate of the potential local maxima placing at the center.
### Return a list of (theta, c) pairs.
def localmax(votes, thetas, cs, thresh, nbhd):
    result = []
    offset = nbhd//2
    votes = np.pad(votes, pad_width=offset, mode='constant', constant_values=0)
    
    for t in range(offset, offset + thetas.shape[0]):
        for c in range(offset, offset + cs.shape[0]):
            if votes[t][c] > thresh and votes[t][c] == find_nbhd_max(votes, offset, t, c):
                result.append((thetas[t - offset], cs[c - offset]))
    return result

def find_nbhd_max(votes, offset, t, c):
    nbhd_matrix = votes[t - offset : t + offset, c - offset : c + offset]
    return np.max(nbhd_matrix)


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
   
    
    
