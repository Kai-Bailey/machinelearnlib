import numpy as np
from . import plots

def gradientDescent(mlModel):
    weights = mlModel.weights
    features = mlModel.features
    labels = mlModel.labels
    reg = mlModel.regularization
    learnRate = mlModel.learningRate

    cost = [mlModel.cost(features, labels, weights, reg)]
    for _ in range(mlModel.iterations):
        grad = mlModel.gradient(features, labels, weights, reg)
        weights -= learnRate * grad
        cost.append(mlModel.cost(features, labels, weights, reg))

    # Save the cost for each iteration for ploting
    mlModel.cost = cost
    return mlModel

def gradientDescentNeuralNetwork(mlModel):
    weights = mlModel.weights
    features = mlModel.features
    labels = mlModel.labels
    testFeatures = mlModel.testFeatures
    testLabels = mlModel.testLabels
    reg = mlModel.regularization
    learnRate = mlModel.learningRate

    testCost = [mlModel.cost(testFeatures, testLabels, weights, reg)]
    cost = [mlModel.cost(features, labels, weights, reg)]
    for i in range(mlModel.iterations):
        grad = mlModel.backPropagation(features, labels, weights, reg)
        
        for j in range(len(weights)):
            weights[j] -= learnRate * grad[j]

        currCost = mlModel.cost(features, labels, weights, reg)
        cost.append(currCost)
        currTestCost = mlModel.cost(testFeatures, testLabels, weights, reg)
        testCost.append(currTestCost)
        print("Iteration:{} | Cost:{} | TestCost: {}".format(i, cost[i], testCost[i]))

    mlModel.testCost = testCost
    mlModel.cost = cost
    return mlModel

def accuracyNeuralNetworkTest(mlModel):
    features = mlModel.testFeatures
    weights = mlModel.weights
    labels = mlModel.testLabels

    numEx = features.shape[0]
    predictions = mlModel.predict(features, weights)

    correct = 0
    for i in range(len(predictions)):
        if labels[predictions[i], i] == 1:
            correct += 1

    print("Test Set Accuracy: ", correct/numEx)
    return mlModel

def accuracyNeuralNetwork(mlModel):
    features = mlModel.features
    labels = mlModel.labels
    testFeatures = mlModel.testFeatures
    testLabels = mlModel.testLabels
    weights = mlModel.weights

    testNumEx = testFeatures.shape[0]
    numEx = features.shape[0]
    testPredictions = mlModel.predict(testFeatures, weights)

    testCorrect = 0
    for i in range(len(testPredictions)):
        if testLabels[testPredictions[i], i] == 1:
            testCorrect += 1

    predictions = mlModel.predict(features, weights)

    for i in range(10):
        print("Prediction: ", predictions[i])
        plots.plotMNIST(features[i,:])

    correct = 0
    for i in range(len(predictions)):
        if labels[predictions[i], i] == 1:
            correct += 1

    print("Accuracy: ", correct/numEx)
    print("Test Set Accuracy: ", testCorrect/testNumEx)
    return mlModel