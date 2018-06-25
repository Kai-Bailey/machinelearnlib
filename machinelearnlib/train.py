from models import logisticRegression as logReg
import models.linearRegression 
import numpy as np


def gradientDescent(mlModel):
    model = mlModel["model"]
    weights = mlModel["weights"]
    features = mlModel["features"]
    labels = mlModel["labels"]
    reg = mlModel["regularization"]
    learnRate = mlModel["learningRate"]

    cost = [model.cost(features, labels, weights, reg)]
    for i in range(100):
        grad = model.gradient(features, labels, weights, reg)
        weights -= learnRate * grad
        cost.append(model.cost(features, labels, weights, reg))

    