import numpy as np

def initializeWeights(mlModel):
    """
    Randomly initalize weights in range [-randInitRange, randInitRange]. If randInitRange is not defined 
    it will default to 0.1.
    """

    randInitRange = mlModel.randInitRange

    numFeatures = mlModel.features.shape[1]
    mlModel.weights = np.random.rand(numFeatures+1) * 2 * randInitRange - randInitRange

    return mlModel

    # TODO neural net initialization