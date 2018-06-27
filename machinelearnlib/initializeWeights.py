import numpy as np

def initializeWeights(mlModel):
    """
    Randomly initalize weights in range [-randInitRange, randInitRange]. If randInitRange is not defined 
    it will default to 0.1.
    """

    if 'randInitRange' in mlModel:
        randInitRange = mlModel['randInitRange']
    else:
        randInitRange = 0.1

    numFeatures = mlModel['features'].shape[1]
    mlModel['weights'] = np.random.rand(numFeatures+1) * 2 * randInitRange - randInitRange


    # TODO neural net initialization