import numpy as np
from .. import loadModel
from .. import activationFunc

def cost(features, labels, weights, reg):
    """
    Compute the cost for the neural network based on the current weights.

    :param features: Numpy matrix of input data used to make prediction.
                      Each row is a training example and each feature is a column.
    :param weights: Numpy array of the learned weights of the model.
    :param labels: Numpy matrix of one hot vectors. Each column is a training example. 
    :param reg: Integer regularization hyperparameter

    :returns: An int which is the cost of the neural net.
    """

    cost = 0    
    numEx = features.shape[0]

    _, activ = forwardPropagation(features, weights)
    hyp = activ[-1]

    # Compute cost without regularization
    cost = (-labels * np.log(hyp)) - ((1-labels) * np.log(1-hyp))
    cost = np.sum(np.sum(cost, axis=1))
    cost = cost * (1/numEx)

    # Compute regularization cost
    # Note this implementation does not regularize the bias units
    reg_cost = 0
    for weight in weights:
        test = np.sum(np.square(weight[:,1:]))
        reg_cost += test
    reg_cost = (reg / (2*numEx)) * reg_cost

    return cost + reg_cost


def forwardPropagation(features, weights):
    """
    Given the input data features and the weights computes the predictions from the neural network.

    :param features: Numpy matrix of input data used to make prediction.
                      Each row is a training example and each feature is a column.
    :param weights: Numpy array of the learned weights of the model.
    :returns: Two lists of numpy arrays containing the intermidiate value (weighted sum before activation func) and the
              activation of the neural network for each layer except for the input layer. Each column is a training
              example and each row is a node in the neural net.
    """
    
    # If only one example is provided
    if features.ndim == 1:
        features = features.reshape(1, features.shape[0])


    num_ex = features.shape[0]
    bias_collumn = np.ones((num_ex, 1))
    activ = np.concatenate((bias_collumn, features), axis=1)
    activ = np.transpose(activ)

    activations = []        # After passing through activation function
    intermediates = []      # Weighted sum of inputs before activation function
    for weight in weights:  
        inter = np.dot(weight, activ)
        intermediates.append(inter)
        activ = activationFunc.sigmoid(inter)
        activations.append(activ)

        # Add column of ones for bias unit
        bias_row = np.ones((1,num_ex))
        activ = np.concatenate((bias_row, activ), axis=0)

    return intermediates, activations


def predict(features, weights):
    """
    Given the input data features and the weight calls forward propogation and returns the index of value
    the neural network is most confident in.

    :param features: Numpy matrix of input data used to make prediction.
                      Each row is a training example and each feature is a column.
    :param weights: Numpy array of the learned weights of the model.
    :returns: The predicted value from the neural network for each training example.
    """

    _, activations = forwardPropagation(features, weights)
    return np.argmax(activations[-1],axis=0)


def backPropagation(features, labels, weights, reg):
    """
    Computes the gradients of the each weight in the neural network.

    :param features: Numpy matrix of input data used to make prediction.
                      Each row is a training example and each feature is a column.
    :param weights: Numpy array of the learned weights of the model.
    :param labels: Numpy matrix of one hot vectors. Each column is a training example. 
    :param reg: Integer regularization hyperparameter
    :returns: A list of numpy arrays where each numpy array is the gradient for a set of weights.
              Matches the shape of mlModel.weights.
    """
    numEx = features.shape[0]
    numFeat = features.shape[1]

    # Build list of numpy arrays to store the cumulative gradient for each weight
    # Gradient is not stored for the bias node 
    accumGrad = []
    for weight in weights:
        accumGrad.append(np.zeros(weight.shape))

    # Iterate through each training example
    for i in range(numEx):
        errors = []
        intermediate, activations = forwardPropagation(features[i,:], weights)
        errors.append(activations[-1] - labels[:,i].reshape(labels.shape[0], 1))

        # Compute errors for each layer of nodes from right to left
        for weight, inter in zip(reversed(weights[1:]), reversed(intermediate[:-1])):
            err = np.dot(weight[:,1:].T, errors[-1]) * activationFunc.sigmoid_gradient(inter)
            errors.append(err)

        # Reverse errors so it nodes mach shape of neural net (left to right)
        errors.reverse()        

        # Add te bias unit
        currFeature = np.hstack((1, features[i,:])).reshape(1, numFeat+1)

        # Calculate gradient for each weight in neural network
        accumGrad[0] += np.dot(errors[0], currFeature)
        for accum, error, activ in zip(accumGrad[1:], errors[1:], activations[:-1]):
            bias_activ = np.vstack((np.array([1]).reshape((1,1)), activ))
            accum += np.dot(error, bias_activ.T)


    # Divide by number of examples and add gradient of regularization
    for i, weight in zip(range(len(accumGrad)), weights):
        accumGrad[i] = (1/numEx) * accumGrad[i]

        # Don't regularize bias term
        accumGrad[i][:, 1:] += (reg/numEx) * weight[:,1:] 

    return accumGrad



def load(mlModel):
    """
    Load the neural net.
    """
    mlModel.predict = predict
    mlModel.forwardPropagation = forwardPropagation
    mlModel.backPropagation = backPropagation
    mlModel.cost = cost

    loadModel.load("neuralNet")