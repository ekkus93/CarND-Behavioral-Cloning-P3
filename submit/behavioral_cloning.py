# ***Using Tensorflow 1.4.0 and Keras 2.1.2*** 
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Lambda, Conv2D, \
    MaxPooling2D, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
from keras import regularizers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

from random import shuffle, random, randint
import cv2
from skimage import exposure
from tqdm import tqdm
from sklearn.utils import shuffle as shuffle_X_y
from math import pi, cos, sin, degrees, radians, atan2, sqrt
import os

from preprocess import *

def parse_file_name(full_path):
    """
    Get filename without filepath

    Parameters
    ----------
    full_path: str
         Full path of a file

    Returns
    -------
    str:
         filename without filepath
    """
    if '/' in full_path:
        return full_path.split('/')[-1]
    else:
        return full_path

def load_data(data_dir):
    """
    Read driving_log.csv from data_dir into pandas DataFrame.

    Parameters
    ----------
    data_dir: str
         Path of directory where driving_log.csv is.

    Returns
    -------
    pandas DataFrame:
         The contents of driving_log.csv with image file names.
    """
    colnames = ['center_img', 'left_img', 'right_img', 'steering_angle', 
                'throttle', 'break', 'speed']
    driving_log_pd = pd.read_csv('%s/driving_log.csv' % data_dir, sep=',', names=colnames)
    
    for colname in ['center_img', 'left_img', 'right_img']:
        driving_log_pd[colname] = [parse_file_name(x) for x 
                                   in driving_log_pd[colname].tolist()]
        
    return driving_log_pd

def display_images(X, start_idx=0, end_idx=None,  columns = 5, use_gray=False, 
                   apply_fnc=None, figsize=(32,18)):
    """
    Display a set of images

    Parameters
    ----------
    X: numpy array of images
         Images to be displayed
    start_idx: int
         Start index for images
    end_idx: int
         End index for images
    columns: int
         Number of columns of images
    use_gray: bool
         True for RGB images.  False for grayscale images.
    apply_fnc: function
         An function to apply to each image before displaying.
    figsize: tuple of int
         Display height and width of images.
    """
    if end_idx is None:
        end_idx = X.shape[0]
        
    if apply_fnc is None:
        apply_fnc = lambda image: image
        
    plt.figure(figsize=figsize)

    num_of_images = end_idx - start_idx
    rows = num_of_images / columns + 1
    
    for i in range(start_idx, end_idx):
        image = X[i]
        
        _i = i % num_of_images
        plt.subplot(rows, columns, _i + 1)
        
        if use_gray:
            plt.imshow(apply_fnc(image), cmap="gray")
        else:
            plt.imshow(apply_fnc(image)) 
            
    plt.tight_layout()
            
    plt.show()

def read_imgs(img_dir, file_names):
    """
    Read list of images from disk.

    Parameters
    ----------
    img_dir: str
         Directory of images.
    file_names: list of str
         List of image file names.

    Returns
    -------
    numpy array of images:
         Images from disk.
    """
    img_arr = []
    
    for file_name in file_names:
        img = imread('%s/%s' % (img_dir, file_name))
        img_arr.append(img)
        
    return np.stack(img_arr)

def split_train_test(img_steering_pd, train_perc=0.7, val_perc=0.2):
    """
    Randomly split data into train/val/test sets.

    Parameters
    ----------
    img_steering_pd: pandas DataFrame
         DataFrame with center_img and steering_angle.

    Returns
    -------
    train_pd: pandas DataFrame
         Training center_imgs and steering_angles.
    val_pd: pandas DataFrame
         Validation center_imgs and steering_angles.
    test_pd: pandas DataFrame
         Test center_imgs and steering_angles.
    """
    idx_len = len(img_steering_pd.index)
    idxs = list(range(idx_len))
    shuffle(idxs)
    
    idx1 = int(idx_len*train_perc)
    idx2 = idx1 + int(idx_len*val_perc)
    
    train_pd = img_steering_pd.iloc[idxs[:idx1]]
    val_pd = img_steering_pd.iloc[idxs[idx1:idx2]]
    test_pd = img_steering_pd.iloc[idxs[idx2:]]
    
    return train_pd, val_pd, test_pd

def make_model(input_shape = (80, 160, 3), num_fully_conn=512, p = 0.5, l=1e-4, alpha=0.3):
    """
    Make VGG-like model with Keras.

    Parametrs
    ---------
    input_shape: tuple of int
        Tuple of height, width and channels.
    num_fully_conn: int
        Number of nodes in the final connected layer.
    p: float
        Dropout keep percentage
    l: float
        Coefficient for L2 regularization.
    alpha: float
        Alpha for leaky RELU.

    Returns
    -------
    Keras model:
       VGG-like model.
    """
    model = Sequential()

    # conv block 1
    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same', 
                     activation=None, input_shape=input_shape,
                     kernel_regularizer=regularizers.l2(l)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation=None,
              kernel_regularizer=regularizers.l2(l)))
    model.add(LeakyReLU(alpha=alpha))    
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    
    # conv block 2
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation=None,
                     kernel_regularizer=regularizers.l2(l)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation=None,
                     kernel_regularizer=regularizers.l2(l)))
    model.add(LeakyReLU(alpha=alpha))    
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))


    # conv block 3
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation=None,
                     kernel_regularizer=regularizers.l2(l)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation=None,
                     kernel_regularizer=regularizers.l2(l)))
    model.add(LeakyReLU(alpha=alpha))    
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    
    model.add(Dropout(p))    
    model.add(Flatten())          

    # fully conn block 1
    model.add(Dense(num_fully_conn, activation=None, kernel_regularizer=regularizers.l2(l)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(p))
    
    model.add(Dense(1))
    
    return model

def flip_imgs(imgs):
    """
    Flip images.

    Parameters
    ----------
    imgs: numpy array of images
          Images to flip.

    Returns
    -------
    numpy array of images:
          Flipped images with the same dimensions as imgs.
    """
    flip_img_arr = [np.fliplr(imgs[i]) for i in range(imgs.shape[0])]

    return np.stack(flip_img_arr)

def flip_y(y):
    """
    Flips steering angles.

    Parameters
    ----------
    y: numpy array
        Steering angles.

    Returns
    -------
    numpy array:
        All steering angles multiplied by -1.0.
    """
    return -y

def image_gen(X_files, y, batch_size, img_dir, size=(80, 160)):
    """
    Generator for dynamically creating training images.

    Parameters
    ----------
    X_files: list of str
         List of image files.
    y: numpy array
         Corresponding steering angles for each image in X_files.
    batch_size: int
         Number of images for each batch.
    img_dir: string
         Directory of image files.
    size: tuple of int
         Height and width for resizing images.

    Returns
    -------
    curr_X: numpy array of images
         Batch of images. The number of images is the same as the batch_size. 
    curr_y: numpy array
         Corresponding steering angles for curr_x. 
    """
    X_len = len(X_files)
    idxs = list(range(X_len))
    
    while True:
        shuffle(idxs)
        y = y[idxs]
        X_files = [X_files[i] for i in idxs]
        
        for i in range(0, X_len, batch_size):
            end_idx = i+batch_size
            
            if end_idx > X_len:
                continue   
  
            curr_y = y[i:end_idx]
            curr_X_files = X_files[i:end_idx]
            curr_X = read_imgs(img_dir, curr_X_files)

            if randint(0,1):
                # flip training data
                curr_X = flip_imgs(curr_X)
                curr_y = flip_y(curr_y)
            
            curr_X = preprocess_images(curr_X, size=size, apply_normalize=True)

            yield curr_X, curr_y

def train_model(model, X_train_files, y_train, img_dir, X_val, y_val, callbacks, size=(80,160),
                batch_size=32, lr=0.0001, epochs=10, workers=1, verbose=0):
    """
    Trains model.

    Parameters
    ----------
    model: Keras Model
         Model to be trained.
    X_train_files: list of str
         List of training files.
    y_train: numpy array
         Corresponding training steering angles for each image.
    img_dir: str
         Directory with training images.
    X_val: numpy array of images
         Validation images.
    y_val: numpy array
         Corresponding validation steering angles for each image.
    callback: list of Keras Callbacks
         Callbacks for fit_generator().
    size: tuple of int
         Height and width of resized images for image generator.
    batch_size: int
         Training batch size.
    lt: float
         Learning rate.
    epochs: int
         Number of training epochs.
    workers: int
         Number of workers for image generator.
    verbose: int
         Verbose setting for fit_generator().
    """
    assert len(X_train_files) == y_train.shape[0]
    assert len(X_train_files) > 0

    optimizer = Adam(lr=lr)
    model.compile(loss='mse', optimizer=optimizer)

    train_gen = image_gen(X_train_files, y_train, batch_size, img_dir, size=size)

    if len(X_train_files) < batch_size:
        steps_per_epoch = 1
    else:
        steps_per_epoch = int(len(X_train_files)/batch_size)
    assert steps_per_epoch > 0
    
    model.fit_generator(train_gen, validation_data=(X_val, y_val), 
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs, workers=workers, verbose=verbose,
                        use_multiprocessing=True, callbacks=callbacks)   
    
    return model
