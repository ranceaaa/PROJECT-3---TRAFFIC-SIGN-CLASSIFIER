# PROJECT-3---TRAFFIC-SIGN-CLASSIFIER

[//]: # (Image References)

[image1]: ./images/hist_train.jpg "Visualization Training Data Set"
[image2]: ./images/hist_valid.jpg "Visualization Validation Data Set"
[image3]: ./images/25.jpg "Image Visualization"
[image4]: ./images/normalized.jpg "Normalized"
[image5]: ./images/augmented.jpg "Augmented"
[image6]: ./images/hist_train_new.jpg "Visualization New Training Data Set"
[image7]: ./images/hist_valid_new.jpg "Visualization New Validation Data Set"
[image8]: ./test_img/30.jpg "Traffic Sign 1"
[image9]: ./test_img/no.jpg "Traffic Sign 2"
[image10]: ./test_img/priority.jpg "Traffic Sign 3"
[image11]: ./test_img/road_work.jpg "Traffic Sign 4"
[image12]: ./test_img/stop.jpg "Traffic Sign 5"
[image13]: ./test_img/stop2.jpg "Traffic Sign 6"
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

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

---

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique.

I tried different preprocessing techniques, but the best results were accomplished using RGB images with a normalization applied. I also went forward with this option because 
considering the signs have different colors, a RGB image is more usefull. Here is an example of a normalized image data. 

![alt text][image4]

After I tried the model for a few times I decided that I need to generate additional data. In this repository few datasets can be found. For example *new_x_train_data_3.p* has around 230.000 images and the goal of this data set was to get close to a relatively equal distribution. *new_x_train_data_6.p* used a different approach, creating data only by working with the pixel values and it contains around 100.000 images. In the notebook, I used *new_x_train_data_5.p* , the new data is composed by: zoom-in of the original image and left shift and right shift of the image. Also, for each one of the new training data set, a new validation data set has been made using the same functions. In the pictures below the new data set can be seen and you can compare the distribution with the original one.

![alt text][image5]

![alt text][image6]

![alt text][image7]

I started creating more datasets because I had some problems with Tensorflow 2.0 and I tought the new data set is the problem, but the real problem was the placeholder.
So, feel free to use any dataset.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x18 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x72  		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x72				|
| Fully connected |  outputs 1800  |
| Dropout |  0.3 | 
| Fully connected		| outputs 240   			|
| Fully connected   | outputs 148         |
| Dropout | 0.3  |
| Fully connected   | outputs 43          |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters :

* Batch size = 128
* Learning rate = 0.0006
* Epochs = 70
* Beta = 0.0005

Adam Optimizier and L2 regularization were used.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.955 
* test set accuracy of 0.961

If an iterative approach was chosen:
* What was the first architecture that was tried? * LeNet5
* What were some problems with the initial architecture? * The initial architecture wasn't able to pass 0.88 on validation set.
* How was the architecture adjusted and why was it adjusted? The number of filters has been changed and due to this, all the fully connected layers dimensions have been changed.
Also, two dropout layers were added. The last addition is L2 regularization.
* What are some of the important design choices and why were they chosen? The choice that improved the model the most was the addition of the dropout layers ( preventing overfitting). 


### Test a Model on New Images

#### 1. Choose 6 traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

![alt text][image8] ![alt text][image9] ![alt text][image10] 
![alt text][image11] ![alt text][image12] ![alt text][image13]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30     		| Speed limit (30km/h)  									| 
| road_work  			| General caution 										|
| no				| No entry									|
| stop2	      		| Stop				 				|
| stop		| Stop 							|
| priority		| Priority road							|

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%.

#### 3. Describe how certain the model is when predicting on each of the 6 new images by looking at the softmax probabilities for each prediction.

In the notebook I've used top 7, but here I will show only top 5 of the softmax probabilities.

For the first image, we got a probability of 1.00 and it was correct. The top five softmax probabilities can be seen below.

* First image - Speed limit (30km/h)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (30km/h)   									| 
| .00     				| Speed limit (20km/h)										|
| .00					| Speed limit (70km/h)											|
| .00	      			| Speed limit (50km/h)				 				|
| .00				    | Speed limit (80km/h)     							|


* Second image - Road work

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .4998         			| General caution  									| 
| .4970     				| Children crossing									|
| .0014					| Pedestrians											|
| .006	      			| Speed limit (30km/h)				 				|
| .006				    | Bicycles crossing						|


* Third image - No entry

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999         			| No entry 									| 
| .0001 				| Stop							|
| .00				| Speed limit (20km/h)										|
| .00	      			| No passing		 				|
| .00				    | Bicycles crossing				|

* Forth image - Stop

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop   									| 
| .00     				| No entry										|
| .00					| Bicycles crossing											|
| .00	      			| Speed limit (30km/h)				 				|
| .00				    | Speed limit (80km/h)     							|


* Fifth image - Stop

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9987       			| Stop  									| 
| .0012     				| Bicycles crossing												|
| .0001 					| No entry											|
| .00 	      			| Priority road			 				|
| .00 				    | Road Work				|


* Sixth image - Priority

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00       			| Priority road 									| 
| .00 				| No entry							|
| .00				| Stop										|
| .00	      			| Bicycles crossing			 				|
| .00				    | Traffic signals		|

The results can be seen also in the notebook displayed in charts.

### Pay attention to Tensorflow, mostly if using both v1 and v2 methods.
