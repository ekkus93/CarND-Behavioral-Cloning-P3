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

from random import shuffle, random
import cv2
from skimage import exposure
from tqdm import tqdm
from sklearn.utils import shuffle as shuffle_X_y
from math import pi, cos, sin, degrees, radians, atan2, sqrt
import os

from preprocess import *

def parse_file_name(full_path):
    if '/' in full_path:
        return full_path.split('/')[-1]
    else:
        return full_path

def load_data(data_dir, smooth_angle=False):
    colnames = ['center_img', 'left_img', 'right_img', 'steering_angle', 
                'throttle', 'break', 'speed']
    driving_log_pd = pd.read_csv('%s/driving_log.csv' % data_dir, sep=',', names=colnames)
    
    for colname in ['center_img', 'left_img', 'right_img']:
        driving_log_pd[colname] = [parse_file_name(x) for x 
                                   in driving_log_pd[colname].tolist()]

    if smooth_angle:
        driving_log_pd = smooth_steering_angle(driving_log_pd)
        
    return driving_log_pd

def display_images(X, start_idx=0, end_idx=None,  columns = 5, use_gray=False, 
                   apply_fnc=None, figsize=(32,18)):
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
    img_arr = []
    
    for file_name in file_names:
        img = imread('%s/%s' % (img_dir, file_name))
        img_arr.append(img)
        
    return np.stack(img_arr)

def split_train_test(img_steering_pd, train_perc=0.7, val_perc=0.2):
    idx_len = len(img_steering_pd.index)
    idxs = list(range(idx_len))
    shuffle(idxs)
    
    idx1 = int(idx_len*train_perc)
    idx2 = idx1 + int(idx_len*val_perc)
    
    train_pd = img_steering_pd.iloc[idxs[:idx1]]
    val_pd = img_steering_pd.iloc[idxs[idx1:idx2]]
    test_pd = img_steering_pd.iloc[idxs[idx2:]]
    
    return train_pd, val_pd, test_pd


def make_model(input_shape = (80, 160, 3), num_fully_conn=512, p = 0.5, weight_decay=1e-4, alpha=0.3):
    model = Sequential()

    # conv block 1
    model.add(Conv2D(8, (3, 3), strides=(1, 1), padding='same', 
                     activation=None, input_shape=input_shape,
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(8, (3, 3), strides=(1, 1), padding='same', activation=None,
              kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(LeakyReLU(alpha=alpha))    
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    
    # conv block 2
    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation=None,
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation=None,
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(LeakyReLU(alpha=alpha))    
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
            
    # conv block 3
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation=None,
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation=None,
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(LeakyReLU(alpha=alpha))    
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    # conv block 4
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation=None,
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation=None,
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(LeakyReLU(alpha=alpha))    
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    # conv block 5
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation=None,
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation=None,
                     kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(LeakyReLU(alpha=alpha))    
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    
    model.add(Dropout(p))    
    model.add(Flatten())          

    # fully conn block 1
    model.add(Dense(num_fully_conn, activation=None, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(p))
    
    model.add(Dense(1))
    
    return model

def image_gen(X_files, y, batch_size, img_dir, size=(80, 160)): 
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
            
            curr_X = preprocess_images(curr_X, size=size)

            yield curr_X, curr_y

def train_model(model, X_train_files, y_train, img_dir, X_val, y_val, callbacks, size=(80,160),
                batch_size=32, lr=0.0001, epochs=10, workers=1, verbose=0):
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

def weight_steering_angle(steering_angles, weights):
    assert len(steering_angles) == len(weights)

    num = sum([steering_angles[i]*weights[i] for i in range(len(steering_angles))])
    denom = sum(weights)

    return num/denom

def smooth_steering_angle(img_steering_pd):
    _img_steering_pd = img_steering_pd.copy()

    mean_steering_angle_arr = []
    weights = [0.125, 0.25, 0.5, 4.0, 0.5, 0.25, 0.125]
    for i in range(len(_img_steering_pd.index)):
        steering_angles = [0.0] * len(weights)

        for j in range(-3, 3):
            if i+j >= 0 and i+j < len(_img_steering_pd.index):
                row = _img_steering_pd.iloc[i]
                steering_angles[j] = row['steering_angle']

        mean_steering_angle_arr.append(weight_steering_angle(steering_angles, weights))
    
    _img_steering_pd['mean_steering_angle'] = mean_steering_angle_arr
    _img_steering_pd = _img_steering_pd.drop('steering_angle', 1)
    _img_steering_pd.rename(mapper={'mean_steering_angle': 'steering_angle'},
                                    axis='columns', inplace=True)
    
    return _img_steering_pd
