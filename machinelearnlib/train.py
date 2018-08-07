import numpy as np
import plots

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

def backPropogation(mlModel):

    activation = mlModel.forwardPropagation(mlModel.features, mlModel.weights)
    print(activation[-1][:][0])

    pred = mlModel.predict(mlModel.features, mlModel.weights)
    print(pred)

    return mlModel