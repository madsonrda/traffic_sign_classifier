# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/countimg.png "Count plot"
[image2]: ./images/signs.png "random signs"
[image3]: ./images/5newimg.png "New images"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Python and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The bar chart showing the count of each traffic sign present in the train set.

![alt text][image1]

Here is a visualization of random sample images of each traffic sign class.

![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I tried to balance the train set by generating random samples of the minor class images and rotating them randomly, I also tried to convert to grayscale and normalize all images . However, when I built the model,  I didn't achieve greater accuracy than just normalizing all images. Therefore, in my preprocessing step I only perform the normalization 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I start from using the Lenet model from the lesson. And my final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten				| output 400									|
| RELU					|												|
| Fully connected		| output 120   									|
| RELU					|												|
| Fully connected		| output 84   									|
| RELU					|												|
| Dropout					|											|
| Fully connected		| output 43   									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 40 EPOCHs, BATCH_SIZE equal to 128, learning rate of 0.001 and a dropout of 0.25

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95%
* test set accuracy of 94%

My first approach was test the same Lenet from the lesson, but the accuracy was not good enough on validation (0.871). Then, I tried adding/removing some dropout layers with different combinations of EPOCH and dropout. So, I achieved an accuracy about 95% on validation set.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] 

The road work sign image is distorted and a bit inclined and rotated, so the model could have difficult to predict it. The other images have few distortions and are easier to predict.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| No entry   									| 
| Priority road 		| Priority road									|
| Bumpy road			| Bumpy road									|
| Speed limit (50km/h)	| Speed limit (50km/h)			 				|
| Stop 					| Stop      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is quite sure that this is a No entry sign (probability of 0.98), and the image does contain a Road work sign. So, the model is completly wrong. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| No entry   									| 
| .01     				| Stop 										|
| .00					| Traffic signals								|
| .00	      			| Beware of ice/snow			 				|
| .00				    | Speed limit (20km/h)      							|


For the second image, the model is sure that this is a Priority road (probability of 1.0), and the image does contain a Priority road sign.  The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority road   									| 
| 0     				| End of all speed and passing limits 										|
| 0					| Speed limit (30km/h)								|
| 0	      			| End of no passing by vehicles over 3.5 metric tons			 				|
| 0				    | Right-of-way at the next intersection      							|

For the third image, the model is sure that this is a Bumpy road sign (probability of 1.0), and the image does contain a Bumpy road sign.  The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Bumpy road   									| 
| 0     				| Bicycles crossing 										|
| 0					| No vehicles								|
| 0	      			| Traffic signals			 				|
| 0				    | Stop      							|

For the fourth image, the model is quite sure that this is a Speed limit (50km/h) sign (probability of .99), and the image does contain a Speed limit (50km/h) sign.  The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (50km/h)   									| 
| 0     				| Speed limit (30km/h) 										|
| 0					| Speed limit (80km/h)								|
| 0	      			| Speed limit (100km/h)			 				|
| 0				    | 10,No passing for vehicles over 3.5 metric tons      							|

For the fourth image, the model is quite sure that this is a Speed limit (50km/h) sign (probability of .99), and the image does contain a Speed limit (50km/h) sign.  The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (50km/h)   									| 
| 0     				| Speed limit (30km/h) 										|
| 0					| Speed limit (80km/h)								|
| 0	      			| Speed limit (100km/h)			 				|
| 0				    | 10,No passing for vehicles over 3.5 metric tons      							|

For the fith image, the model is sure that this is a Stop sign (probability of 1.0), and the image does contain a Stop sign.  The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop   									| 
| 0     				| No entry 										|
| 0					| Speed limit (30km/h)								|
| 0	      			| Speed limit (60km/h)			 				|
| 0				    | Road work      							|


