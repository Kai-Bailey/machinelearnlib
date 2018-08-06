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

def neuralNetInitializeWeights(mlModel):
    """
    Initialize weights of the neural network based on the netArchitechture and the randInitRange.
    """

    randInitRange = mlModel.randInitRange
    netArchitecture = mlModel.netArchitechture
    weights = []
    
    for i in range(len(netArchitecture)-1):
        weight = np.random.rand(netArchitecture[i+1], netArchitecture[i]+1) * 2 * randInitRange - randInitRange       
        weights.append(weight)

    mlModel.weights = weights
