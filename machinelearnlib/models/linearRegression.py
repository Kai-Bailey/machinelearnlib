import numpy as np

def cost(features, labels, weights, regularization=0):
    """
    Computes the squared error between the labels and predicted output with regularization for each weight.

    Param: features - Numpy matrix of input data used to make prediction.
                      Each row is a training example and each feature is a column.
           labels - Numpy column vector of the correct output for each training example.
           weights - Numpy column vector of the learned weights of the model.
           regularization - Integer representing the regularization.

    Return: Cost of the linear regression model.
    """

    pass


def predict(features, weights):
    """
    Given the input data features and the weights computes the predictions from the linear regression model.

    Param: features - Numpy matrix of input data used to make prediction.
                      Each row is a training example and each feature is a column.
           weights - Numpy column array of the learned weights of the model.
    """
    
    # Prepend a column of ones for bias feature
    numEx = features.shape()[0]
    biasFeature = np.ones((numEx,1))
    features = np.concatenate((biasFeature, features))

    prediction = np.dot(features, weights)

    return prediction


def gradient(features , labels, weights):
    """
    Computes the gradient of the cost function For linear regression. 

    Param: features - Numpy matrix of input data used to make prediction.
                      Each row is a training example and each feature is a column.
    labels - Numpy column vector of the correct output for each training example.
    weights - Numpy column vector of the learned weights of the model.
    """
    pass


