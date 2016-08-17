# Neural Network For Digit Prediction (Machine Learning)
Creates and trains a neural network to predict images (28x28 pixels) of hand-written digits.  
The accuracy based on its own training examples is then computed and printed to the log.

Training example records in the mnist set consist of 29 elements separated by ","  
Element 0 is the labeled class and the remaining 28 elements are pixel intensities for each image pixel.

The default structure of the neural network consists of:
* 28 input nodes for each image pixel (calculated)
* 1 hidden layer comprised of 30 nodes (hard-coded)
* 10 classes for hand-written digits (calculated)
