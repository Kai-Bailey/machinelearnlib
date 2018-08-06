import numpy as np
import loadModel

def cost(features, labels, weights, reg=0):
    """
    Computes the squared error between the labels and predicted output with regularization for each weight.

    :param features: Numpy matrix of input data used to make prediction. Each row is a training example and each feature is a column.
    :param labels: Numpy array of the correct output for each training example.
    :param weights: Numpy array of the learned weights of the model.
    :param reg: Integer representing the regularization.

    :return: Cost of the linear regression model.
    """

    # Number of training set examples
    numEx = features.shape[0]

    # Compute the squared Error
    pred_labels= predict(features, weights)
    squaredError = (pred_labels - labels)**2
    cost = (1/(2 * numEx)) * squaredError.sum()

    # Add the Regularization for all weights except for the bias
    cost += (reg/(2*numEx)) * (weights[1:]**2).sum()

    return cost


def predict(features, weights):
    """
    Given the input data features and the weights computes the predictions from the linear regression model.

    Param: features - Numpy matrix of input data used to make prediction.
                      Each row is a training example and each feature is a column.
           weights - Numpy array of the learned weights of the model.
    Returns Numpy array of with a prediction for each example.
    """
    
    # Prepend a column of ones for bias feature
    numEx = features.shape[0]
    biasFeature = np.ones((numEx,1))
    features = np.concatenate((biasFeature, features), axis=1)

    prediction = np.dot(features, weights.T)

    return prediction


def gradient(features , labels, weights, reg=0):
    """
    Computes the gradient of the cost function For linear regression. 

    :param features: Numpy matrix of input data used to make prediction. Each row is a training example and each feature is a column.
    :param labels: Numpy array of the correct output for each training example.
    :param weights: Numpy array of the learned weights of the model.

    :return: A numpy array containing the partial derivative of the cost function for each weight.
    """
    
    numEx = features.shape[0]
    prediction = predict(features, weights)

    # Prepend a column of ones for bias feature
    numEx = features.shape[0]
    biasFeature = np.ones((numEx,1))
    features = np.concatenate((biasFeature, features), axis=1)

    # Compute the gradient
    difference = (prediction - labels)
    grad = (1/numEx)*np.dot(features.T,difference.T)
    
    # Add the gradient of the regualrization term except for bias
    grad[1:] = grad[1:] + (reg/numEx)*weights[1:]

    
    return grad

def load(mlModel):
    """
    """

    mlModel.cost = cost
    mlModel.gradient = gradient
    mlModel.predict = predict
    loadModel.load("linearRegression")

    # mlModel["cost"] = cost
    # mlModel["gradient"] = gradient
    # mlModel["predict"] = predict
    # mlModel["process_list"] = ["loadData.loadData(mlModel)", "plots.plotTrainDataLinearRegression(mlModel)", "initializeWeights.initializeWeights(mlModel)",
    #                             "featureScaling.meanNormalization(mlModel)", "featureScaling.stdNormalization(mlModel)", "train.gradientDescent(mlModel)", "plots.plotCost(mlModel)"]