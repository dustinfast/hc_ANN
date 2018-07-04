# Handwriting Classification with Aritifical Neural Networks

This application learns to classifify handwritten english letters by building
an Artifical Neural Network (ANN) model from a collection of pre-labeled handwritten characters.  
The ANN is a 3D Matrix of fully-connected sigmoid neurons. Uses error back propogation and a logistic activation function. Some ANN properties may be adjusted as specified below.

##Data 

Training and validation data is located in ./dataset. Each row represents a letter and properties of a handwrittten representation of it. It was obtained from https://archive.ics.uci.edu/ml/datasets/Letter+Recognition and divided into two disjoint sets (training and validation).

**Extraction**  
Data is first extracted from each file in ./dataset and pushed into vectors, then objects of type SigmoidDataRow are created for each row of data. Input parameters are normalized ( 0 <= param <=  1).

**Learning**  
The sigmoid is trained, for each row of training data, with error backpropagation, LEARNING_ITERATIONS times.
After all learning iterations, the model is validated against each row of validation data. A confusion matrix is then displayed with accuracy results.

See inline source documentation for more information.

## Performance

The model was trained seperately with varying values of LEARNING_ITERATIONS and LEARNING_RATE.
BIAS was held at a constant -1 and all weights were initialized to 0.5. Results are located in Analysis.docx

## Usage

To compile: g++ -std=c++11 main.cpp

## Constants

The following constants in main.cpp may be use to adjust ANN properties:

* LEARNING_ITERATIONS
* LEARNING_RATE
* BIAS
* BIAS_WEIGHT
* NETWORK_LAYERS
* NETWORK_LAYER_COUNT
* TRAINING_DATAFILE
* VALIDATION_DATAFILE






