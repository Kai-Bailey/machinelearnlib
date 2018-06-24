from models import logisticRegression as logReg
import numpy as np

if __name__ == "__main__":
    features = np.array([[2, 5, 4], [3, 6, 8], [9, 3, 2]])
    weights = np.array([0.04, 0.01, 0.02, 0.03])
    labels = np.array([1, 0, 1])

    J  = logReg.cost(features, labels, weights)

