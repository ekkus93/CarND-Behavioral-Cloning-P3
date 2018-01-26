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
batch_size = 32
workers=1

def load_data(data_dir):
    train_pd_file = '%s/train.p' % data_dir
    val_pd_file = '%s/val.p' % data_dir
    test_pd_file = '%s/test.p' % data_dir

    if not os.path.exists(train_pd_file) or not os.path.exists(val_pd_file) or not os.path.exists(test_pd_file):
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

def main():
    driving_log_pd = load_data(data_dir)
    img_dir = '%s/IMG' % data_dir

    # ## Data
    train_pd, val_pd, test_pd = load_data(data_dir)

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
    
    # ## Model
    model = make_model(input_shape = (64, 320, 3), p = 0.5, weight_decay = 1e-4)
    print(model.summary())
    print()
    
    model_graph_file = '%s/model.png' % data_dir
    plot_model(model, to_file=model_graph_file)

    # ## Train Model (primary data)
    epochs=10
    lr = 0.00001
    weight_decay = 1e-4
    verbose = 2
    
    model = train_model(model, X_train_files, y_train, img_dir, X_val, y_val,
                        lr=lr, epochs=epochs, workers=workers, verbose=verbose)

    test_loss = model.evaluate(X_test, y_test, verbose=verbose)
    print("test loss: %3f" % test_loss)

    model.save('%s/model.h5'%data_dir)
    print("=======================================================")
    print()

if __name__ == "__main__":
    main()
