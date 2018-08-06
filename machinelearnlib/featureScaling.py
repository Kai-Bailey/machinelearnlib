import numpy as np

def meanNormalization(mlModel):
    """
    Subtract the mean from each feature from that feature so the data
    is centered around 0.

    :param mlModel: Class defining the users options for the ml model.
    :return mlModel: Class defining the users options for the ml model. 
    """

    features = mlModel.features
    numFeatures = features.shape[1]

    means = []
    for i in range(numFeatures):
        mean = np.mean(features[:,i], dtype='float32')
        features[:, i] = features[:, i] - mean
        means.append(mean)

    # Save to do predictions on real data later
    mlModel.mean = np.asarray(means) 

    return mlModel

def stdNormalization(mlModel):
    """
    Divide each feature by the standard deviation for that feature.

    :param mlModel: Class defining the users options for the ml model.
    :return mlModel: Class defining the users options for the ml model.  
    """

    features = mlModelfeatures
    numFeatures = features.shape[1]

    stds = []
    for i in range(numFeatures):
        std = np.std(features[:, i], dtype='float32')
        features[:, i] = np.divide(features[:, i], std)
        stds.append(std)

    # Save std to do predictions on real data later
    mlModel.std = np.asarray(stds)

    return mlModel

def randomShuffle(mlModel):
    """
    Randomly shuffles all rows of the training and testing data.

    :param mlModel: Class defining the users options for the ml model.
    :return mlModel: Class defining the users options for the ml model. 
    """

    # Shuffle training data
    shuffledIndex = np.random.permutation(len(mlModel.labels))
    mlModel.features = mlModel.features[shuffledIndex]
    mlModel.labels = mlModel.labels[shuffledIndex]

    # Shuffle testing data
    shuffledIndex = np.random.permutation(len(mlModel.labels))
    mlModel.features = mlModel.features[shuffledIndex]
    mlModel.labels = mlModel.labels[shuffledIndex]

    return mlModel