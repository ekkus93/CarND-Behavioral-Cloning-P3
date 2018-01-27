# ***Using Tensorflow 1.4.0 and Keras 2.1.2*** 
import tensorflow as tf
import keras

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

data_dir = 'data/data_primary'
extra_data_dir = 'data/data_extra'
batch_size = 128
workers=7

def load_split_data(data_dir):
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

def preprocess_Xy_data(Xy_pd, img_dir, crop_x0=0, crop_y0=48, crop_x1=None, crop_y1=112):
    _X = preprocess_images(read_imgs(img_dir, Xy_pd['center_img'].tolist()),
                           x0=crop_x0, y0=crop_y0, x1=crop_x1, y1=crop_y1)
    _y = np.array(Xy_pd['steering_angle'])

    return _X, _y

def predict_from_files(model, img_dir, X_files, batch_size=32,
                       crop_x0=0, crop_y0=48, crop_x1=None, crop_y1=112):
    y_hat_arr = []

    for i in range(0, len(X_files), batch_size):
        curr_X_files = [X_files[j] for j in range(i, min(i+batch_size, len(X_files)))]

        curr_X = read_imgs(img_dir, curr_X_files)
        curr_X = preprocess_images(curr_X, x0=crop_x0, y0=crop_y0, x1=crop_x1, y1=crop_y1)
        
        curr_y_hat = model.predict(curr_X, batch_size=curr_X.shape[0])
        y_hat_arr.append(curr_y_hat)

    y_hat =  np.concatenate(y_hat_arr, axis=0)
    assert y_hat.shape[0] == len(X_files)

    return y_hat

def filter_incorrect(model, X_files, y, img_dir, perc_err=0.05, batch_size=32):
    y_hat = predict_from_files(model, img_dir, X_files)

    err_np = np.abs(y-y_hat) <= perc_err
    keep_idxs = [i for i in range(err_np.shape[0]) if err_np[i] <= perc_err]

    return [X_files[i] for i in keep_idxs], y[keep_idxs]

def main():
    driving_log_pd = load_data(data_dir)
    img_dir = '%s/IMG' % data_dir

    # ## Data
    train_pd, val_pd, test_pd = load_split_data(data_dir)

    # ### Preprocessing Images

    # The images will be preprocessed with the following steps:
    # 1. Crop
    #     * The top and bottom of the images will be cropped to reduce the size of the input.
    # 2. Normalization
    #     * The images are normalized with a range of -0.5 to 0.5.
    X_train_files = train_pd['center_img'].tolist()
    y_train = np.array(train_pd['steering_angle'])

    X_val, y_val = preprocess_Xy_data(val_pd, img_dir, crop_x0=0, crop_y0=48, crop_x1=None, crop_y1=112)
    X_test, y_test = preprocess_Xy_data(val_pd, img_dir, crop_x0=0, crop_y0=48, crop_x1=None, crop_y1=112)

    cnt = int(0.1*len(X_train_files))
    X_train_files = X_train_files[:cnt]
    y_train = y_train[:cnt]

    cnt = int(0.1*X_val.shape[0])
    X_val = X_val[:cnt]
    y_val = y_val[:cnt]    
    
    # ## Model
    input_shape = (64, 320, 3)
    p = 0.5
    weight_decay = 1e-4
    alpha = 0.01
    
    model = make_model(input_shape = input_shape, p = p, weight_decay = weight_decay,
                       alpha =alpha)
    print(model.summary())
    print()
    
    model_graph_file = '%s/model.png' % data_dir
    plot_model(model, to_file=model_graph_file)

    # ## Train Model (primary data)
    epochs=5
    lr = 0.001
    weight_decay = 1e-4
    verbose = 2

    # pretrain
    cnt = int(0.1*len(X_train_files))
    _X_train_files = X_train_files[:cnt]
    _y_train = y_train[:cnt]

    model = train_model(model, _X_train_files, _y_train, img_dir, X_val, y_val,
                        lr=lr, epochs=epochs, workers=workers, verbose=verbose)

    for i in range(5):
        _X_train_files, _y_train = filter_incorrect(model, X_train_files, y_train, img_dir, perc_err=0.05,
                                                    batch_size=batch_size)
        print("percent not learned: %3f" % _y_train.shape[0]/y_train.shape[0])
        
        cnt = int(0.1*len(_X_train_files))
        _X_train_files = _X_train_files[:cnt]
        _y_train = _y_train[:cnt]
    
        model = train_model(model, _X_train_files, _y_train, img_dir, X_val, y_val,
                            lr=lr, epochs=epochs, workers=workers, verbose=verbose)

    test_loss = model.evaluate(X_test, y_test, verbose=verbose)
    print("test loss: %3f" % test_loss)
    
    model.save('%s/model.h5' % data_dir)
    print("=======================================================")
    print()

if __name__ == "__main__":
    main()
