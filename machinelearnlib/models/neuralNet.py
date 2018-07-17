import numpy as np
import copy
from matplotlib import pyplot as plt


def initializeWeights(dimensions, bound):
    '''
    Initialize the weights to uniformly dstributed random values from -bound to
    +bound.
    '''

    weights = []

    for i in range(0, dimensions.shape[0] - 1):
        weights.append(np.random.uniform(-bound, bound, (dimensions[i + 1],
                                                         dimensions[i] + 1)))

    return weights


def shuffle(features, labels):
    '''
    Randomly shuffles the features and labels while making each feature is
    lined up with the corresponding label.
    '''

    index = np.arange(features.shape[0])
    np.random.shuffle(index)

    return features[index], labels[index]


def sigmoid(array):
    '''
    All elements of array will go through the sigmoid function.
    '''

    array = 1 / (1 + np.e ** (-array))
    return array


def forwardPropagation(feature, weights, dimension):
    '''
    Forward propagate through the neural newtork using the current weights and
    and biases. The activation of the nodes in the second to last layers will
    be returned.
    '''

    feature = np.insert(feature, 0, 1)

    activation = []   # List of np arrays containing activations for each node
    activation.append(sigmoid(np.matmul(weights[0], feature)))
    activation[0] = np.insert(activation[0], 0, 1)

    for i in range(1, dimensions.shape[0] - 1):
        activation.append(sigmoid(np.matmul(weights[i], activation[i - 1])))
        activation[i] = np.insert(activation[i], 0, 1)

    activation[-1] = np.delete(activation[-1], 0)

    return activation


def prediction(features, weights, dimensions):
    '''
    Forward propgate through the neural network using the current weights and
    biases. An array containing one hot vectors for each training example will
    be returned.
    '''

    predictions = np.empty([features.shape[0], dimensions[-1]], dtype=float)
    features = np.insert(features, 0, 1, axis=1)
    features = np.transpose(features)

    for i in range(0, features.shape[1]):
        prevAct = sigmoid(np.matmul(weights[0], features[:, i]))
        prevAct = np.insert(prevAct, 0, 1)

        for j in range(1, dimensions.shape[0] - 2):
            activations = sigmoid(np.matmul(weights[j], prevAct))
            prevAct = np.insert(activations, 0, 1)

        predictions[i, :] = sigmoid(np.matmul(weights[-1], prevAct))

    return predictions


def cost(features, weights, labels, dimensions, regularization):
    '''
    Returns the cost of the current weights and biases on the full training set
    '''

    predictions = prediction(features, weights, dimensions)

    cost = 0.0
    for i in range(0, features.shape[0]):
        for j in range(0, dimensions[-1]):
            if labels[i, j] == 0:
                cost += np.log10(1 - predictions[i, j])
            else:
                cost += np.log10(predictions[i, j])

    cost = -(1 / features.shape[0]) * cost

    regTerm = 0.0
    for i in range(0, len(weights)):
        regTerm += np.sum(weights[i] ** 2)

    regTerm = (regularization / (features.shape[0] * 2)) * regTerm
    cost += regTerm

    return cost


def gradCheck(features, weights, labels, dimensions, reg):
    '''
    gradCheck is used to verify that backPropagation is working correctly and
    should not be used in practice as performance is very slow. Verfifies that
    the gradient in backPropagation is correct by apprroximating the gradient.
    '''

    e = 0.0001
    approxGrad = []

    for l in range(0, len(weights)):
        approxGrad.append(np.empty_like(weights[l]))

        for j in range(weights[l].shape[0]):
            for i in range(weights[l].shape[1]):
                currWeightsP = copy.deepcopy(weights)
                currWeightsP[l][j][i] += e
                currWeightsN = copy.deepcopy(weights)
                currWeightsN[l][j][i] -= e
                costP = cost(features, currWeightsP, labels, dimensions, reg)
                costN = cost(features, currWeightsN, labels, dimensions, reg)
                approxGrad[l][j][i] = (costP - costN) / (2.0 * e)

    print(approxGrad)


def backPropagation(features, weights, labels, dimensions, regularization,
                    learningRate):
    '''
    Computes back propagation through the neural network computing the gradient
    with respect to each weight and bias. The weights and biases are then
    changed accordingly and the new weights are returned.
    '''
    numLayers = dimensions.shape[0]
    numEx = features.shape[0]

    totError = []

    # Create list of error for each weight arranged like weights list
    for i in range(0, len(weights)):
        totError.append(np.empty_like(weights[i]))

    for i in range(0, numEx):
        error = []
        activation = forwardPropagation(features[i], weights, dimensions)

        error.insert(0, activation[-1] - labels[i, :])

        for j in range(numLayers - 2, 0, -1):
            temp = np.matmul(np.transpose(weights[j]), error[0])
            temp2 = np.multiply(temp, activation[j - 1])
            temp3 = np.multiply(temp2, (1 - activation[j - 1]))
            error.insert(0, temp3)

        totError[0] = np.outer(features[i], error[0])
        for j in range(1, numLayers - 1):
            totError[j] += np.matmul(activation[j], error[j])

    grad = []
    newWeights = []
    for i in range(0, numLayers - 1):
        grad.append((1 / numEx) + regularization * weights[i])
        newWeights.append(weights[i] - learningRate * grad[i])

    print(grad)
    return newWeights


def optimize(features, weights, labels, dimensions, reg, learningRate,
             iterations):
    '''
    Iterates through backPropagation and displays a graph plotting the cost
    function versus the number of iterations.
    '''

    costFunc = [cost(features, weights, labels, dimensions, reg)]
    for i in range(0, iterations):
        weights = backPropagation(features, weights, labels, dimensions, reg,
                                  learningRate)
        costFunc.append(cost(features, weights, labels, dimensions, reg))

    plt.figure()
    plt.title("Cost Function vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.plot(costFunc)
    plt.show()

    return weights