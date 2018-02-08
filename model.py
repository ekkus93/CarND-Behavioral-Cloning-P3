# ***Using Tensorflow 1.4.0 and Keras 2.1.2*** 
import tensorflow as tf
import keras

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

from random import shuffle, random
import cv2
from skimage import exposure
from tqdm import tqdm
from sklearn.utils import shuffle as shuffle_X_y
from math import pi, cos, sin, degrees, radians, atan2, sqrt
import os

from preprocess import *
from behavioral_cloning import *

data_dir = 'data'

def load_split_data(data_dir):
    """
    Randomly splits data into train/validation/test and save pickle files of them.
    If the pickle files already exist, load the pickle files instead of resplitting the data. 

    Parameters
    ----------
    data_dir: str
        The data directory must contain the following file:
        * driving_log.csv
    
    Returns
    -------
    train_pd : pandas DataFrame
        Training Data
    val_pd : pandas DataFrame
        Validation Data
    test_pd : pandas DataFrame
        Test Data
    """
    train_pd_file = '%s/train.p' % data_dir
    val_pd_file = '%s/val.p' % data_dir
    test_pd_file = '%s/test.p' % data_dir

    if not os.path.exists(train_pd_file) or not os.path.exists(val_pd_file) or not os.path.exists(test_pd_file):
        driving_log_pd = load_data(data_dir)
        
        train_pd, val_pd, test_pd = split_train_test(driving_log_pd)
        train_pd.to_pickle(train_pd_file)
        val_pd.to_pickle(val_pd_file)
        test_pd.to_pickle(test_pd_file)
    else:
        train_pd = pd.read_pickle(train_pd_file)
        val_pd = pd.read_pickle(val_pd_file)
        test_pd = pd.read_pickle(test_pd_file)    

    return train_pd, val_pd, test_pd

def preprocess_Xy_data(Xy_pd, img_dir, size=(80, 160)):
    """
    Preprocesses image and steering angle data for training.

    Parameters
    ----------
    Xy_pd: pandas DataFrame
         Contains file names for raw images and steering angles
    img_dir: str
         The directory of the raw images
    size: tuple
         A tuple of height and width

    Returns
    -------
    _X: numpy array
         Preprocessed images
    _y: numpy array
         Coresponding steering angles

    """
    _X = preprocess_images(read_imgs(img_dir, Xy_pd['center_img'].tolist()), size=size, apply_normalize=True)
    _y = np.array(Xy_pd['steering_angle'])

    return _X, _y

def predict_from_files(model, img_dir, X_files, size=(80,160), batch_size=32):
    """
    Predict steering angle a list of image files

    Parameters
    ----------
    model: Keras Model
         A trained model for predicting the steering angle from an image
    img_dir: string
         The directory of the images
    X_files: list of str
         A list of image file names in img_dir
    size: tuple
         A tuple of height and width for resizing the images during preprocessing

    Returns
    -------
    y_hat: numpy array
         An array of predicted steering angles.
    """
    y_hat_arr = []

    for i in range(0, len(X_files), batch_size):
        curr_X_files = [X_files[j] for j in range(i, min(i+batch_size, len(X_files)))]

        curr_X = read_imgs(img_dir, curr_X_files)
        curr_X = preprocess_images(curr_X, size=size, apply_normalize=True)
        
        curr_y_hat = model.predict(curr_X, batch_size=curr_X.shape[0])
        y_hat_arr.append(curr_y_hat)

    y_hat =  np.concatenate(y_hat_arr, axis=0)
    assert y_hat.shape[0] == len(X_files)

    return y_hat

def main():
    input_shape = (32, 32, 4)
    img_dir = '%s/IMG' % data_dir

    # ## Data
    train_pd, val_pd, test_pd = load_split_data(data_dir)

    # ### Preprocessing Images
    X_train_files = train_pd['center_img'].tolist()
    y_train = np.array(train_pd['steering_angle'])

    X_val, y_val = preprocess_Xy_data(val_pd, img_dir, size=input_shape[:2])
    X_test, y_test = preprocess_Xy_data(val_pd, img_dir, size=input_shape[:2])
    
    # ## Model
    model_file = '%s/model.h5' % data_dir

    # model parameters
    batch_size = 32
    workers=7

    num_fully_conn = 256
    p = 0.5
    l = 1e-6
    alpha = 1e-6
    epochs=20
    lr = 1e-4
    verbose = 2

    model = make_model(input_shape = input_shape, num_fully_conn=num_fully_conn,
                       p = p, l = l, alpha =alpha)
        
    print(model.summary())
    print()

    checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=verbose,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=1)
    callbacks = [checkpoint]
    
    assert X_val.shape[1] == input_shape[0], X_val.shape[1]
    assert X_val.shape[2] == input_shape[1], X_val.shape[2]
        
    # ## Train Model 
    model = train_model(model, X_train_files, y_train, img_dir, X_val, y_val, callbacks, size=input_shape[:2],
                        lr=lr, epochs=epochs, workers=workers, verbose=verbose)

    # load best model
    model = load_model(model_file)
    test_loss = model.evaluate(X_test, y_test, verbose=verbose)
    print("test loss: %3f" % test_loss)

    # clear the session manually
    K.clear_session()
    
if __name__ == "__main__":
    main()
