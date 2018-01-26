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

    """
    ### ***Subset data for debugging***
    subset_factor = 0.05
    X_train_subset_cnt = int(subset_factor*len(X_train_files))
    X_train_files = X_train_files[:X_train_subset_cnt]
    y_train = y_train[:X_train_subset_cnt]

    X_val_subset_cnt = int(subset_factor*X_val.shape[0])
    X_val = X_val[:X_val_subset_cnt]
    y_val = y_val[:X_val_subset_cnt]

    X_test_subset_cnt = int(subset_factor*X_test.shape[0])
    X_test = X_test[:X_test_subset_cnt]
    y_test = y_val[:X_test_subset_cnt]
    """
    
    # ## Model
    model = make_model(input_shape = (64, 320, 3), p = 0.5)

    model_graph_file = '%s/model.png' % data_dir
    plot_model(model, to_file=model_graph_file)

    # ## Train Model (primary data)
    print("### Primary Model")
    epochs=20
    model = train_model(model, X_train_files, y_train, img_dir, X_val, y_val, epochs=epochs, workers=workers)

    test_loss = model.evaluate(X_test, y_test)
    print("test loss: %3f" % test_loss)

    model.save('%s/model.h5'%data_dir)
    print("=======================================================")
    print()

    # ## Fine tune model with extra data
    # The first model has problems right after the bridge.  It doesn't bank left hard enough and it ends up driving off
    # the road. Train the model with extra data from just that section of the track. The learning rate of 0.0001 will be
    # used, same as the first model.  All of the extra data will be used for training. The validation data from the first
    # model will be reused.
    model_file = '%s/model.h5' % data_dir

    print("### Fine Tuned Model with Extra Data")
    epochs=20
    model2 = fine_tune_model_train(extra_data_dir, model_file, X_val, y_val, epochs=epochs, workers=workers)

    test_loss2 = model2.evaluate(X_test, y_test)
    print("test loss: %3f" % test_loss)

    model2_file = '%s/model.h5' % extra_data_dir
    model2.save(model2_file)

if __name__ == "__main__":
    main()
