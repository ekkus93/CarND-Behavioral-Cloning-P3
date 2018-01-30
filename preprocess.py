import numpy as np
import cv2
from skimage import exposure
from scipy.misc import imresize
from line import *

def crop_image(img, x0=0, y0=0, x1=None, y1=None):
    if x1 is None:
        x1 = img.shape[1]
        
    if y1 is None:
        y1 = img.shape[0]
    
    return img[y0:y1, x0:x1, :]

def apply_histogram_equalization(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized_image = exposure.equalize_adapthist(gray_image)
    
    return (255.0*equalized_image).astype(np.uint8)

def resize(img, size=(32, 32)):
    return imresize(img, size=size)

def normalize(img):
    return img / 255.0 - 0.5

def preprocess_image(img, size=(80, 160), apply_normalize=False):
    _img = img
    
    #_img = weighted_img(find_lane3(_img), _img)
    _img = grayscale(find_lane3(_img))

    _img = resize(_img, size=size)

    if apply_normalize:
        _img = normalize(_img)

    if len(_img.shape) == 2:
        _img = _img.reshape(list(_img.shape)+[1])

    return _img

def preprocess_images(X, size=(32, 32),
                      convert_to_rgb=False, apply_normalize=True):
    return np.array([preprocess_image(X[i], size=size, apply_normalize=apply_normalize) for i in range(X.shape[0])])

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def crop_roi(img):
    """
    Crops region of interest for an image.  The cropped region is a preset trapezoidal shape
    based on the dimensions of the image.
    
    Parameters
    ----------
    img : numpy image array
        grayscale image

    Returns
    -------
    numpy image array
        cropped image based on img
    """
    bottom_y = int(0.8*img.shape[0])
    top_y = int(0.3*img.shape[0])
    top_left_x = int(0.2*img.shape[1])
    top_right_x = int(0.8*img.shape[1])
    bottom_left_x = 0
    bottom_right_x = img.shape[1]

    pt0 = [bottom_left_x, bottom_y]
    pt1 = [top_left_x, top_y]
    pt2 = [top_right_x, top_y]
    pt3 = [bottom_right_x, bottom_y]
    
    bounding_box = np.array([pt0, pt1, pt2, pt3], np.int32)
    
    cropped_img = region_of_interest(img, [bounding_box])   
    
    return cropped_img

def find_lane3(image):
    """
    Like find_lane() but uses hough_lines3() instead of hough_lines().
    
    Parameters
    ----------
    image: numpy image array
        image of road
        
    Returns
    -------
    numpy image array
        image with black background with lane lines
    """    
    gray = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    masked_edges = crop_roi(edges)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 50    # minimum number of votes (intersections in Hough grid cell)
    
    min_line_length = 3 #minimum number of pixels making up a line
    max_line_gap = 40    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    color_edges = hough_lines3(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    
    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)   

    return lines_edges 

def draw_lines3(img, lines, color=[255, 0, 0], thickness=2):
    """
    Draw lines on original image.  The lines will be separated by left and right lines.
    Horizontal lines will be filtered out. The means of the left and right lines will 
    be used to draw the lines on the original image.
    
    Parameters
    ----------
    img: numpy image array
        original image
    lines: list of (x1,y1,x2,y2)
        list of line points
    color: array of int
        RGB values for drawn lines
    thickness: int
        thickness of the drawn lines
    """    
    line_objs = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            curr_line_obj = Line(x1,y1,x2,y2)
            line_objs.append(curr_line_obj)
    
    left_line_objs = []
    right_line_objs = []
    for line_obj in line_objs:
        if abs(line_obj.m) > 0.15:
            if line_obj.m > 0.0:
                left_line_objs.append(line_obj)
            else:
                right_line_objs.append(line_obj)
            
    left_line_list = LineCollection(left_line_objs)
    right_line_list = LineCollection(right_line_objs)
    
    bottom_y = img.shape[0]
    
    mean_lines = [left_line_list.get_mean_line_pts(bottom_y), 
                  right_line_list.get_mean_line_pts(bottom_y)]

    for x1,y1,x2,y2 in mean_lines:
        cv2.line(img, (x1, y1), (x2,y2), color, thickness)

def hough_lines3(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines3(line_img, lines, thickness=8)
    return line_img
