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
[image2]: ./imgs/sample_track.png "Sample track"
[image3]: ./imgs/sample_lane_lines.png "Sample lane lines"
[image4]: ./imgs/sample_merged_lane_lines.png "Sample merged lane lines"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Files 

My project includes the following files:
* model.py containing the script to create and train the model
* preprocessing.py containing common functions for preprocessing the image data for model.py and drive.py
* behavioral_cloning.py containing helper functions for model.py
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* model_train.log has the output of the training

#### Running the model

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py data/model.h5
```

#### 3. Submission code 

The model.py file contains the code for training and saving the convolution neural network. The behavioral_modeling.py has functions for creating and training the model.  In addition to this, there are various helper functions.  The pipeline.py has common functions for preprocessing images which are used both by model.py for training the model and in drive.py for the autonomous driving. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create a model to take the original RBG data of a single image with the addition of lane lines to predict a good steering angle.

My first step was to use a convolution neural network model similar to a VGG-style model. I thought this model might be appropriate because it is a pretty simple and reliable convolution neural network that I would build on.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I did the following things adjust the model:
1. Reducing the size of the model
  * Lowering the number of filters in the Conv2D layers
  * Lowering the number of nodes in the Dense layer
2. Adding Dropout
  * Dropout rate was set to 0.5
3. Adding L2 regularization
  * For the Conv2D and Dense layers, the kernel regularizers were set to use L2 regularization with an l of 1e-6
  
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My model consists of a VGG-style convolution neural network. (behavioral_cloning.py lines 100-142) The model includes leaky RELU layers to introduce nonlinearity. 

| Layer Name       | Layer Type                 | Output Shape | 
|:-----------------|:---------------------------|:-------------|
| conv2d_1         | Conv2D                     | (32, 32, 8)  |     
| leaky_re_lu_1    | LeakyReLU                  | (32, 32, 8)  |
| conv2d_2         | Conv2D                     | (32, 32, 8)  |   
| leaky_re_lu_2    | LeakyReLU                  | (32, 32, 8)  |    
| max_pooling2d_1  | MaxPooling2D               | (16, 16, 8)  |       
| conv2d_3         | Conv2D                     | (16, 16, 16) |     
| leaky_re_lu_3    | LeakyReLU                  | (16, 16, 16) |       
| conv2d_4         | Conv2D                     | (16, 16, 16) |      
| leaky_re_lu_4    | LeakyReLU                  | (16, 16, 16) |       
| max_pooling2d_2  | MaxPooling2D               | (8, 8, 16)   |       
| conv2d_5         | Conv2D                     | (8, 8, 32)   |       
| leaky_re_lu_5    | LeakyReLU                  | (8, 8, 32)   |       
| conv2d_6         | Conv2D                     | (8, 8, 32)   |        
| leaky_re_lu_6    | LeakyReLU                  | (8, 8, 32)   |       
| max_pooling2d_3  | MaxPooling2D               | (4, 4, 32)   |        
| dropout_1        | Dropout                    | (4, 4, 32)   |       
| flatten_1        | Flatten                    | (512)        |        
| dense_1          | Dense                      | (256)        |        
| leaky_re_lu_7    | LeakyReLU                  | (256)        |        
| dropout_2        | Dropout                    | (256)        |        
| dense_2          | Dense                      | (1)          |    

![][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 4-5 laps on track one using center lane driving.  I thought that this was enough data for the model to learn to average out the best steering angle for the track.  Here's a sample of the training images:

![][image2]

The total number of images collected was 18644.  The split of the data is:

| Type       | Count   |
|:-----------|--------:|
| Train      | 13050   |
| Validation | 3728    |
| Test       | 1866    |

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (behavioral_cloning.py line 169).  The initial learning rate was 1e-4.

#### 4. Appropriate training data

Training data consists of recordings of driving the car around the track 4-5 times. 

Originally, I extracted the lane lines using a Canny filter and a Hough transform and merged them with the original image:
![][image4]

I ended up adding the lane lines as a separate layer so the model didn't need to learn how to extract the merged lane lines from the image.
![][image3]













### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

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

#### Pipeline

All of the images go through a preprocessing pipeline for training and automous driving of the car.  The code for preprocessing pipeline function, preprocess_images(), is in preprocess.py. Preprocessing involves the following steps:
1. Adding lane lines layer
  * In addition to using the RGB color layers of the image, an extra layer is added with lane lines.  The lane lines are extracted from a grayscale image of the original image using a canny filter and a hough transform.
2. Resizing
  * All layers are resized to 32x32.
3. Normalization
  * All layers are centered around 0.0 with a range of -0.5 to 0.5.
