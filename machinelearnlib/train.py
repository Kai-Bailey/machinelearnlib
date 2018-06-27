from models import logisticRegression as logisticRegression
import models.linearRegression as linearRegression
import numpy as np
import plots

def gradientDescent(mlModel):
    model = mlModel["model"]
    weights = mlModel["weights"]
    features = mlModel["features"]
    labels = mlModel["labels"]
    reg = mlModel["regularization"]
    learnRate = mlModel["learningRate"]

    cost = [linearRegression.cost(features, labels, weights, reg)]
    for i in range(10):
        grad = linearRegression.gradient(features, labels, weights, reg)
        weights -= learnRate * grad
        cost.append(linearRegression.cost(features, labels, weights, reg))

        plots.plotPrediction(mlModel)

