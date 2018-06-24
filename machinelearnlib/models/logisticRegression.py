import numpy as np
import activationFunc


def cost(features, labels, weights, reg=0):
    """
    Computes the squared error between the labels and predicted output with regularization for each weight.

    Param: features - Numpy matrix of input data used to make prediction.
                      Each row is a training example and each feature is a column.
           labels - Numpy array of the correct output for each training example.
           weights - Numpy array of the learned weights of the model.
           reg - Integer representing the regularization.

    Return: Cost of the linear regression model.
    """
    # Number of examples in the training set
    numEx = features.shape[0]

    # Compute the cost
    pred = predict(features, weights)
    cost  = np.multiply((-1*labels), np.log(pred)) - np.multiply((1-labels), np.log(1-pred))
    cost = (1/numEx)*cost.sum()

    # Add the Regularization for all weights except for the bias
    cost += (reg/(2*numEx)) * (weights[1:]**2).sum()

    return cost


def predict(features, weights):
    """
    Given the input data features and the weights computes the predictions from the logistic regression model.

    Param: features - Numpy matrix of input data used to make prediction.
                      Each row is a training example and each feature is a column.
           weights - Numpy array of the learned weights of the model.
    Returns Numpy array of with a prediction for each example.
    """
    
    # Prepend a column of ones for bias feature
    numEx = features.shape[0]
    biasFeature = np.ones((numEx,1))
    features = np.concatenate((biasFeature, features), axis=1)

    prediction = activationFunc.sigmoid(np.dot(features, weights.T))

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

