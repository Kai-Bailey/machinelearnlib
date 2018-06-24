
mlModel = {
    # Can choose from linearRegression, logisticRegression and neuralNet
    "model":"linearRegression",
    
    # Training data
    # The files containing the features and labels should be placed in the data folder and the 
    # names of the files should be specified below.Features are are arranged so each row is a
    #  training example and each feature is a column.
    "features":"trainFeatures.py",
    "labels":"labels.py",

    # Format of the training data
    # Valid options are numpyArr or csv
    "format":"numpyArr",

    # Hyperparameters
    "learningRate":0.1,
    "regularization":0.1,

    # If the model used is a neural net you can specify the number of nodes in each layer using a list.
    # For example a neural network with an input layer of size 10, hidden layer 12 and output layer of 8
    # would use [10, 12, 8]. The input layer must be the same size as the number of features (columns) in
    # the training data. If you are not using a neural net the list can be left empty.
    "netArchitechture":[]
}

# TODO Feature Scaling options
# TODO Stochastic vs Gradient 
# TODO Randomly Shuffle Data
# TODO Split training set into test, cv and train (% split?)