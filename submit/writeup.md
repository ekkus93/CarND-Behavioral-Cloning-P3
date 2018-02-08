# **Behavioral Cloning** 

## Behavioral Cloning Project

Based on the mac_sim driving simulator, a model was built to predict steering angles from images to drive the car. The training data was collected from simulator by driving the car manually around the track.  From the training data, the behavior of manually driving the car was cloned by the model.

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
[image5]: ./imgs/sample_cropped_track.png "Sample cropped track"
[image6]: ./imgs/training_losses.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
## Files 

My project includes the following files:

* *model.py* containing the script to create and train the model
* *preprocessing.py* containing common functions for preprocessing the image data for model.py and drive.py
* *behavioral_cloning.py* containing helper functions for *model.py*
* *line.py* containing Line models for Hough transform in *preprocessing.py*
* *preprocess.py* containing functions for preprocessing images of the track
* *drive.py* for driving the car in autonomous mode
* *model.h5* containing a trained convolution neural network
* *writeup.md* summarizing the results
* *model_train.log* has the output of the training.
* *run1.mp4* is a video of driving around the track autonomously.

## Running the model

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py data/model.h5
```

## Submission code 

The model.py file contains the code for training and saving the convolution neural network. The *behavioral_modeling.py* has functions for creating and training the model.  In addition to this, there are various helper functions.  The *pipeline.py* has common functions for preprocessing images which are used both by *model.py* for training the model and in *drive.py* for the autonomous driving. 

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create a model to take the original RBG data of a single image with the addition of lane lines to predict a good steering angle.

My first step was to use a VGG-style convolution neural network model. I thought this model might be appropriate because it is a pretty simple and reliable convolution model that I could build upon.  By adding the lane lines with the preprocessing step, I didn't think it was necessary to use a more complicated model such as the NVIDIA model. Since the lane lines were already extracted in the preprocessing step, the model didn't need work as hard trying to learn how to extract the lane line features from the original images on its own.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my initial models had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. Also, when driving the car in autonomous mode with these initial models, the car would frequently cross over the lane lines or drive off the track.  

To combat the overfitting, I did the following things adjust the model:

1. Adding Dropout
    * Dropout rate was set to 0.5.
2. Adding L2 regularization
    * The kernel regularizers were set to use L2 regularization with an l of 1e-6 for the Conv2D and Dense layers.
3. Data Augmentation
    * Additional training data was created by flipped the training images and the steering angle. 
  
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### 2. Final Model Architecture

My model consists of a VGG-style convolution neural network similar to what was described here in the Keras documentation under ["VGG-like convnet"](https://keras.io/getting-started/sequential-model-guide/). The model includes leaky RELU layers to introduce nonlinearity. I chose leaky RELU's over regular RELU's to try to avoid the dying RELU problem.

| Layer Name       | Layer Type                 | Output Shape | 
|:-----------------|:---------------------------|:-------------|
| conv2d_1         | Conv2D                     | (32, 32, 16)  |     
| leaky_re_lu_1    | LeakyReLU                  | (32, 32, 16)  |
| conv2d_2         | Conv2D                     | (32, 32, 16)  |   
| leaky_re_lu_2    | LeakyReLU                  | (32, 32, 16)  |    
| max_pooling2d_1  | MaxPooling2D               | (16, 16, 16)  |       
| conv2d_3         | Conv2D                     | (16, 16, 32) |     
| leaky_re_lu_3    | LeakyReLU                  | (16, 16, 32) |       
| conv2d_4         | Conv2D                     | (16, 16, 32) |      
| leaky_re_lu_4    | LeakyReLU                  | (16, 16, 32) |       
| max_pooling2d_2  | MaxPooling2D               | (8, 8, 32)   |       
| conv2d_5         | Conv2D                     | (8, 8, 64)   |       
| leaky_re_lu_5    | LeakyReLU                  | (8, 8, 64)   |       
| conv2d_6         | Conv2D                     | (8, 8, 64)   |        
| leaky_re_lu_6    | LeakyReLU                  | (8, 8, 64)   |       
| max_pooling2d_3  | MaxPooling2D               | (4, 4, 64)   |        
| dropout_1        | Dropout                    | (4, 4, 64)   |       
| flatten_1        | Flatten                    | (1024)        |        
| dense_1          | Dense                      | (256)        |        
| leaky_re_lu_7    | LeakyReLU                  | (256)        |        
| dropout_2        | Dropout                    | (256)        |        
| dense_2          | Dense                      | (1)          |    

![][image1]

### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 4-5 laps on track one using center lane driving.  I thought that this was enough data for the model to learn to average out the best steering angle for the track.  Here's a sample of the training images:

![][image2]

The total number of images collected was 18644.  The split of the data is:

| Type       | Count   |
|:-----------|--------:|
| Train      | 13050   |
| Validation | 3728    |
| Test       | 1866    |

### 4. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (behavioral_cloning.py line 169).  The initial learning rate was 1e-4.

The other model parameters were tuned by trial and error. Different parameter configurations were run for 10 epochs with one tenth of the data. The parameters for the model with the best validation loss were chosen.

| Parameter                                 | Variable Name  | Value       |
|:------------------------------------------|:---------------|------------:|
| Input Image Shape                         | input_shape    | (32, 32, 4) |
| Batch Size                                | batch_size     |     32      |
| Number of nodes in fully connected layer  | num_fully_conn |    256      |
| Dropout Keep Percentage                   | p              |    0.5      |
| L2 Regulatization coefficent              | l              |   1e-6      |
| Alpha for Leaky RELU                      | alpha          |   1e-6      |

### 5. Appropriate training data

Training data consists of recordings of driving the car around the track 4-5 times. This seemed to be a good amount of data for the model to use to generalize the track.  If at certain points along the track for a single loop was slightly off from manually driving the car, the model should be able to average what the correct steering angle should be from the laps around the track.

#### Preprocessing Pipeline

All of the training images were preprocessed with the following pipeline:

1. Lane Lines Extraction
2. Image Scaling
3. Normalization

The validation and test images also use the same pipeline.  driver.py was also modified to use the same pipeline as well.

In addition to preprocessing, the train image data were augmented by randomly flipping training images and steering angles.

##### Lane Lines

First the original image of the track was cropped to a region of interest like so:
![][image5]

The region of interest is a trapezoidal shape which narrows on top to the horizon.   

Originally, I extracted the lane lines using a Canny filter and a Hough transform and merged them with the original image.  Below are images with the lane lines merged onto the original images of the track.  This is just to show that the lane lines are lined up well with the actual lane lines of the track. 
![][image4]

Initially, I tried training the model with just the lane lines but there were places along the track which only had one lane line.  Using the lane lines alone wasn't enough information for the model to predict the correct steering angle.  In those cases, the car would just drive off the road after crossing the bridge.  Adding back the RBG layers gives the model more information for the model to try to make a correct prediction for the cases where the lane lines alone wasn't enough.

I ended up adding the lane lines as a separate layer. I could have merged it with the original image of the track above.  Having it as a separate layer simplifies training by not having the model have to learn how to extract the lane lines from the merged image.
![][image3]

##### Image Scaling

The sizes of the original images are 160x320.  During preprocessing, the images were resize down to 32x32.  After experimenting with different sizes, 32x32 still had enough information to train the model and make good predictions while keeping the size of the model relatively small. 

##### Normalization

All pixel values were scaled to the range of -0.5 and 0.5 and centered around 0.0.

#### Data Augmentation

The data was augmented by randomly flipping the image to create a mirror image of the track. The steering angles were also flipped by multipling the steering angles by a -1.0.  Compared to just using the original, unflipped images, his helped the model better generalize the track.  Before data augmentation, the car would sometimes swerve a little outside of the lane lines.  After data augmentation, the driving was much smoother and better centered along the track.

## Training

The model was trained for 20 epochs.  The model with the best validation loss out of the 20 epochs was chosen for the final model.  The final model was from the 20th epoch with a training loss of 0.0225 and validation loss of 0.0236.  The test loss of the final model was 0.0236.

![][image6]

The training loss line oscillate above and below the validation loss line. For the final epoch, the training loss is slightly below the validation loss which might suggest a little bit of overfitting. The model performs well autonomously driving along the track using *drive.py* though.  The validation loss and the test loss for the final model are consistent so we can assume that 0.0236 is a true representation of the loss for the model.

## Conclusion

The final model was tested with *drive.py*.  The model drove the car along the track for about 2 laps.  The video, *run1.mp4*, is a recording of the car driving along the track. The car drove well, centered along the track.  It didn't drive off the road at any point like in earlier models with lower number of training epochs and without data augmentation.

I felt pretty confident that my model did a good job in generalizing driving around the track.  Still, the track that was used to test the final model was the same track where the training data was recorded from.  The model could have memorized parts or all of the track because the same track was used in both cases.  A similar but slightly different track should have been used for the test track.  That would have been a best test for the model.










