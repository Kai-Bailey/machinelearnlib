# machinelearnlib

[![PyPI](https://img.shields.io/pypi/v/machinelearnlib.svg)](https://pypi.org/project/machinelearnlib/)
[![Build Status](https://travis-ci.org/Kai-Bailey/machinelearnlib.svg?branch=master)](https://travis-ci.org/Kai-Bailey/machinelearnlib)

<img src="images/machinelearnlibLarge.jpg" width="55%">

Currently a work in progress.

## Description

Machinelearnlib is a python package that allows user to run different machine learning algorithms. It is designed to be a high level library so users can quickly test different models on their data to find which one works the best. The project was motivated by my desire to learn more about data science and machine learning.

## How to run

Install the library with:

```python
pip install machinelearnlib
```

To use machinelearnlib you must first define the parameters for the model and training data using the model class:

```python
    # Can choose from linearRegression, logisticRegression and neuralNet
    model = "linearRegression"
    
    # Training data
    # The files containing the training data should be placed in the data folder and the 
    # names of the files should be specified below. Features are are arranged so each row is a
    # training example and each feature is a column. Labels should be the last column.
    trainDataFileName = "linearTrain.csv"
    testDataFileName = None #"linearTest.csv"

    # Format of the training data
    # Valid options are csv or numpy array
    fileFormat = "csv"

    # Hyperparameters
    learningRate = 0.0001
    regularization = 0.1
    randInitRange = 0.1
    
    # Number of iterations to run the learning algorithm for
    iterations = 10

    # If the model used is a neural net you can specify the number of nodes in each layer using a list.
    # For example a neural network with an input layer of size 10, hidden layer 12 and output layer of 8
    # would use [10, 12, 8]. The input layer must be the same size as the number of features (columns) in
    # the training data. If you are not using a neural net the list can be left empty.
    netArchitechture = []

    mlModel = Model(model, 
                    trainDataFileName, 
                    fileFormat, 
                    netArchitecture=netArchitechture,
                    learningRate=learningRate, 
                    regularization=regularization, 
                    iterations=iterations,
                    randInitRange=randInitRange)

    run(mlModel)
```

Then, simply import the machinelearnlib and run the model:

```python
import machinelearnlib as ml

ml.run(mlModel)
```

## Data Formatting
Training data must formatted so that each column contains a feature and the last column is the label. 
Each row is a training example.

| feature1 | feature2 | feature3 | label1 |
|----------|----------|----------|--------|
| datapoint1 | datapoint1 | datapoint1 | datapoint1 | 
| datapoint2 | datapoint2 | datapoint2 | datapoint2 | 


## Author
* [Kai Bailey](https://kai-bailey.com) - Software engineering student at the University of Alberta.

## Future Development

* Finish implementing neural network
* Add a support vector machine

## Acknowledgment
Example Data From:

* [Linear Data Set](https://www.kaggle.com/andonians/random-linear-regression)

* [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris)


Thanks to [Coursera](https://www.coursera.org/learn/machine-learning/home/welcome) for the free course!
