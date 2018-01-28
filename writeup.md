# **Behavioral Cloning** 

**Behavioral Cloning Project**

Based on the mac_sim driving simulator, a model was built to predict steering angles from images to drive the car.  The model is based on a [VGG-style convolutional neural network](https://keras.io/getting-started/sequential-model-guide/#examples).  The model was designed to be small enough to train on a cpu in a reasonable amount of time yet still have good performance.  

The training data was collected from simulator by driving the car manually 4 or 5 times around the track.  

The following libraries were used to train the model:
* Tensorflow v1.4.0
* Keras v2.1.2

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./imgs/model.png "Model graph"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### Running the model
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py data/model.h5
```

#### Pipeline

All of the images go through a preprocessing pipeline.  The code for preprocessing pipeline function, preprocess_images(), is in preprocess.py. Preprocessing involves the following steps:
1. Cropping
  * The top and bottom parts of the images are cropped off.  The lane lines in front of the car are the most important feature for predicting the correct steering angle.  The skyline and part of the road at the bottom of the images aren't as essential.
2. Resizing
  * After cropping, the image is rectangular in shape.  The images are resized to a 32x32 square.  This reduces the size of the input considerably. The heights of the images are stretched.  This accentuates the angles of the lane lines
3. Normalization
  * The pixel values are centered around 0.0 for faster training.

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

Two methods were used to reduce overfitting:
1. Dropout 
  * Dropout of p=0.5 was applied to the fully connected block after the activation layers.  
2. L2 regularizers 
  * L2 regularizers with weight_decay=1e-6 were applied the Conv2D and Dense layers.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

| Layer (type)                  | Output Shape                  |
|:------------------------------|:------------------------------|
| conv2d_1 (Conv2D)             | (None, 32, 32, 8)             |
| leaky_re_lu_1 (LeakyReLU)     | (None, 32, 32, 8)             | 
| conv2d_2 (Conv2D)             | (None, 32, 32, 8)             | 
| leaky_re_lu_2 (LeakyReLU)     | (None, 32, 32, 8)             | 
| max_pooling2d_1 (MaxPooling2) | (None, 16, 16, 8)             | 
| conv2d_3 (Conv2D)             | (None, 16, 16, 16)            | 
| leaky_re_lu_3 (LeakyReLU)     | (None, 16, 16, 16)            |
| conv2d_4 (Conv2D)             | (None, 16, 16, 16)            |
| leaky_re_lu_4 (LeakyReLU)     | (None, 16, 16, 16)            | 
| max_pooling2d_2 (MaxPooling2) | (None, 8, 8, 16)              |        
| dropout_1 (Dropout)           | (None, 8, 8, 16)              |
| flatten_1 (Flatten)           | (None, 1024)                  |       
| dense_1 (Dense)               | (None, 320)                   |
| leaky_re_lu_5 (LeakyReLU)     | (None, 320)                   |
| dropout_2 (Dropout)           | (None, 320)                   |
| dense_2 (Dense)               | (None, 1)                     |


#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a convolution neural network model that would be small enough to train in a reasonable amount of time on a cpu.  With the combination of Conv2D and MaPooling layers, I was able to reduce the number of features down to 256 before going to the fully connected layer.

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
