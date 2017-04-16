#**Traffic Sign Recognition** 

##Writeup by Chao Liu


  

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./All-Traffic-Signs.png  "Visualization"
[image2]: ./Number-of-each-Sign.png  "Histogram"
[image3]: ./LeNet_Modified2_Architecture.png  "Architecture"
[image4]: ./traffic-signs-examples/traffic-signs-examples.png "Traffic Signs Examples"
[image5]: ./softmax_problities.png "Softmax Problities"
[image6]: ./Bar_of_Problities.png "Bar Chart of Softmax Problities"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Heiquanjing/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 3th-4th code cell of the IPython notebook.  Here are all classes of the German traffic signs.

![alt text][image1]

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.

For preprocessing, I just only normalized the image data to the range (-1, 1) using the code line X = (X - 128) / 128 . I did NOT grayscale the image and kept the color channels.



####2. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the ninth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling 2x2	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU				|         									|
| Max pooling 2x2		| 2x2 stride,  outputs 5x5x32		|
|Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x400	  	|
|Flatten		    | flatten the layer2(5x5x32->800) and layer3(1x1x400->400)		  	|
|Concatenation	| concatenate the flattened layers to one(800+400=1200) 		|
|RELU				| 										|
|Fully connected		|Inputs 1200, Outputs 200						|
 |RELU				| 										|
|DROPOUT			|keep_prob = 0.5 for training, 1.0 for valid and test	|
|Fully connected		|Inputs 200, Outputs 43						|



####3. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 10th cell of the ipython notebook. 

To train the model, I used an AdamOptimizer method.

####4. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 11-13th cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 96.4% 
* test set accuracy of 94.7%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?   
--- Original Lenet-5.
* What were some problems with the initial architecture?    
--- This model worked  well, but the validation set accuracy was lower than 93%，so I tried to implemente the Sermanet/LeCun model from their traffic sign classifier paper and there was an immediate improvement.  
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.   
--- The architecture of the  modified LeNet: 
1. 5x5 convolution (Input 32x32x3, Output 28x28x16)
2. ReLU
3. 2x2 max pool (Input 28x28x16, Output 14x14x16)
4. 5x5 convolution (Input 14x14x16, Output 10x10x32)
5. ReLU
6. 2x2 max pool (Input 10x10x16, Output 5x5x32)
7. 5x5 convolution (Input 5x5x32, Output 1x1x400)
8. Flatten layers from numbers 7 (1x1x400 -> 400) and 6 (5x5x32 -> 800)
9. Concatenate the two flattened layers to a single size-1200 layer
10. ReLU
11. Dropout layer (keep_prob = 0.5)
12. Fully connected layer (Input 1200, Output 200)
10. ReLU
11. Dropout layer (keep_prob = 0.5)
12. Fully connected layer (Input 200, Output 43) 

![alt text][image3]

* Which parameters were tuned? How were they adjusted and why?   
--- EPOCHES, BATCH_SIZE and the learing rate. The more epoches the high accuracy, but the accuracy will keep in a small range after certain epoches, so too many epoches have no mean. I set 30 to the parameter EPOCHES, but we don't need to wait the model after 30 epoches. Here  I expect the good accuracy will be higher than 0.96, so when I get the valid accuracy larger than 0.96 for 2 consecutive times, the accuracy looks like stable, then I stop training the model.  The BATCH_SIZE is 100. Learning rate is 0.001. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?   
--- The parameter keep_prob for the Dropout is 0.5,  while the count of the combination of net is the most one, and the dropout will improve the overfitting problem significantly.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead Only     			| Ahead Only   									| 
| Children Crossing		| Children Crossing								|
| No Entry				| No Entry										|
| Priority Road			| Priority Road									|
| No Passing			| No Passing	      								|
| Turn Right			| Turn Right									|
| 30 km/h	      			| 30 km/h						 				|
| Yield					| Yield											|
| Go Straight or Left	| Go Straight or Left							|
| Stop					| Stop											|



The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.9%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

Here is the top 5 problities of all ten images:

![alt text][image5]

For the 3rd image, the no entry sign. There is something dirty on this sign, so that maybe affect the prediction. But the model is relatively sure that this is a no entry sign (probability of 1.0). 

For the 9th image the sign is  Go Straight or Left. From the image we can see the sign has a little warp. The model predicted correctly, but with a little lower problities by contrast to other images. 

Here is the softmax problities of the ten images:

![alt text][image6]
