import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pandas as pd

def loadData(mlModel):
    """
    Load in data from the data folder using the file names specified in teh mlModel.

    :param mlModel: Class defining the users options for the ml model.
    :return mlModel: Class defining the users options for the ml model. 
    """

    if mlModel.trainDataFileName != None:
        trainFileName = './data/' + mlModel.trainDataFileName
        trainData = pd.read_csv(trainFileName)
        mlModel.features = np.array(trainData.iloc[:,:-1])
        mlModel.labels = np.array(trainData.iloc[:,-1])

    if mlModel.testDataFileName != None:
        testFileName = './data/' + mlModel.testDataFileName         
        testData = pd.read_csv(testFileName)
        mlModel.testFeatures = np.array(testData.iloc[:,:-1])
        mlModel.testLabels = np.array(testData.iloc[:,-1])

    return mlModel

def loadDataMNIST(mlModel):
    """
    Load the MNIST data set.
    """
    import mnist

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    mlModel.features = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2]))
    
    # Convert labels to one hot
    onehot_train_labels = np.zeros((train_labels.size, 10))
    onehot_train_labels[np.arange(train_labels.size), train_labels] = 1
    mlModel.labels = onehot_train_labels

    mlModel.testFeatures = test_images.reshape((test_images.shape[0], test_images.shape[1] * test_images.shape[2]))

    # Convert labels to one hot
    onehot_test_labels = np.zeros((test_labels.size, 10))
    onehot_test_labels[np.arange(test_labels.size), test_labels] = 1
    mlModel.testLabels = onehot_test_labels

    return mlModel

def loadIrisData(mlModel):
    """
    Load the Iris Data Set from the data folder into the mlModel

    :param mlModel: Class defining the users options for the ml model.
    :return mlModel: Class defining the users options for the ml model. 
    """

    trainFileName = './data/' + mlModel.trainDataFileName
    trainData = pd.read_csv(trainFileName)

    # Convert the labels 
    unique = {}
    uniqueID = 0
    row = 0
    for label in trainData.iloc[:,-1]:
        if label not in unique:
            unique[label] = uniqueID
            uniqueID += 1

        trainData.iloc[row, -1] = unique[trainData.iloc[row, -1]]
        row += 1

    mlModel.features = np.array(trainData.iloc[:,:-1])
    mlModel.labels = np.array(trainData.iloc[:,-1])

    return mlModel


