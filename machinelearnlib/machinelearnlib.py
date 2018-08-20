from . import loadModel
from . import processList

title = """
                  __   _          __                  ___ __ 
  __ _  ___ _____/ /  (_)__  ___ / /__ ___ ________  / (_) / 
 /  ' \/ _ `/ __/ _ \/ / _ \/ -_) / -_) _ `/ __/ _ \/ / / _ \\
/_/_/_/\_,_/\__/_//_/_/_//_/\__/_/\__/\_,_/_/ /_//_/_/_/_.__/

"""
 
class Model():
    """
    User defined class specifing the model and its options.
    """

    def __init__(self, model, trainDataFileName, dataFormat, testDataFileName=None, netArchitecture=None, 
                       learningRate=0.0001, regularization=1, iterations=10, randInitRange=0.1, featureScaling=False):
        self.model = model
        self.trainDataFileName = trainDataFileName
        self.testDataFileName = testDataFileName
        self.netArchitecture = netArchitecture
        self.learningRate = learningRate
        self.regularization = regularization
        self.iterations = iterations
        self.randInitRange = randInitRange
        self.featureScaling = featureScaling

def train(mlModel):
    """
    Cycle through process list defined in loadModel until process list is empty.
    """

    print(title)
    processList.append(loadModel.load_models)
    if mlModel.featureScaling:
        processList.run_after("loadData.loadData")

    args = processList.consume(mlModel)
    while processList.ready():
        args = processList.consume(args)

def main():
    
    # Can choose from linearRegression, logisticRegression and neuralNet
    model = "neuralNet"
    
    # Training data
    # The files containing the training data should be placed in the data folder and the 
    # names of the files should be specified below. Features are are arranged so each row is a
    # training example and each feature is a column. Labels should be the last column.
    trainDataFileName = None
    testDataFileName = None 

    # Format of the training data
    # Valid options are csv or numpy array
    fileFormat = "csv"

    # Hyperparameters
    learningRate = 0.3
    regularization = 0.0001
    randInitRange = 0.1
    
    # Setting featureScaling to True will shuffle the training data, and perform standard deviation 
    # and mean normalization.
    featureScaling = False

    # Number of iterations to run the learning algorithm for
    iterations = 250

    # If the model used is a neural net you can specify the number of nodes in each layer using a list.
    # For example a neural network with an input layer of size 10, hidden layer 12 and output layer of 8
    # would use [10, 12, 8]. The input layer must be the same size as the number of features (columns) in
    # the training data. If you are not using a neural net the list can be left empty.
    netArchitechture = [784, 100, 10]

    mlModel = Model(model, 
                    trainDataFileName, 
                    fileFormat, 
                    netArchitecture=netArchitechture,
                    learningRate=learningRate, 
                    regularization=regularization, 
                    iterations=iterations,
                    randInitRange=randInitRange,
                    featureScaling = featureScaling)

    train(mlModel)


if __name__ == "__main__":
    main()