# PROJECT-3---TRAFFIC-SIGN-CLASSIFIER

[//]: # (Image References)

[image1]: ./examples/hist_train.jpg "Visualization Training Data Set"
[image2]: ./examples/hist_valid.jpg "Visualization Validation Data Set"
[image3]: ./examples/25.jpg "Image Visualization"
[image4]: ./examples/normalized.jpg "Normalized"
[image5]: ./examples/augmented.jpg "Augmented"
[image6]: ./examples/hist_train_new.jpg "Visualization New Training Data Set"
[image7]: ./examples/hist_valid_new.jpg "Visualization New Validation Data Set"
[image8]: ./examples/placeholder.png "Traffic Sign 3"
[image9]: ./examples/placeholder.png "Traffic Sign 4"
[image10]: ./examples/placeholder.png "Traffic Sign 5"

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

Number of training examples = 34799
Number of testing examples = 12630
Number of validation examples = 4410
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the data set.

Here is an exploratory visualization of the data set. The first chart shows how many samples of each class are in the training data set,
the second one also a chart for validation data set and the last picture shows few images from the data set. 

![alt text][image1]

![alt text][image2]

![alt text][image3]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique.

I tried different preprocessing techniques, but the best results were accomplished using RGB images with a normalization applied. I also went forward with this option because 
considering the signs have different colors, a RGB image is more usefull. Here is an example of a normalized image data. 

![alt text][image4]

After I tried the model for a few times I decided that I need to generate additional data. In this repository few datasets can be found. For example *new_x_train_data_3.p* has around 230.000 images and the goal of this data set was to get close to a relatively equal distribution. *new_x_train_data_6.p* used a different approach, creating data only by working with the pixel values and it contains around 100.000 images. In the notebook, I used *new_x_train_data_5.p* , the new data is composed by: zoom-in of the original image and left shift and right shift of the image. In the pictures below the new data set can be seen and you can compare the distribution with the original one.

![alt text][image5]

![alt text][image6]

![alt text][image7]

I started creating more datasets because I had some problems with Tensorflow 2.0 and I tought the new data set is the problem, but the real problem was the placeholder.
So, feel free to use any dataset.






