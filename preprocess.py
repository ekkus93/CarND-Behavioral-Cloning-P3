import numpy as np
import cv2
from skimage import exposure

def crop_image(img, x0=0, y0=0, x1=None, y1=None):
    if x1 is None:
        x1 = img.shape[1]
        
    if y1 is None:
        y1 = img.shape[0]
    
    return img[y0:y1, x0:x1, :]

def normalize(img):
    return img / 255.0 - 0.5

def preprocess_image(img, x0=0, y0=48, x1=None, y1=112, convert_to_rgb=False):
    _img = img
    if not convert_to_rgb:
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    
    _img = crop_image(img, x0, y0, x1, y1)
    
    _img = normalize(_img)
    
    return _img

def preprocess_images(X, x0=0, y0=48, x1=None, y1=112, convert_to_rgb=False):
    _X = np.array([preprocess_image(X[i], convert_to_rgb=convert_to_rgb) for i in range(X.shape[0])])

    #return _X.reshape(list(_X.shape) + [1])
    return _X
