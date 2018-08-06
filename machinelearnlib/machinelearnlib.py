import loadModel
import processList

title = """
                  __   _          __                  ___ __ 
  __ _  ___ _____/ /  (_)__  ___ / /__ ___ ________  / (_) / 
 /  ' \/ _ `/ __/ _ \/ / _ \/ -_) / -_) _ `/ __/ _ \/ / / _ \\
/_/_/_/\_,_/\__/_//_/_/_//_/\__/_/\__/\_,_/_/ /_//_/_/_/_.__/

"""

# TODO Ensure everything is type float
# TODO Reshape feature

def run(mlModel):

    print(title)
    processList.append(loadModel.load_models)

    args = processList.consume(mlModel)
    while processList.ready():
        args = processList.consume(args)


    # print_title()
    # loadData.loadIrisData(mlModel)
    # featureScaling.randomShuffle(mlModel)
    # plots.plotTrainDataLinearRegression(mlModel, 0, 1)
    # initializeWeights.initializeWeights(mlModel)

    # #featureScaling.meanNormalization(mlModel)
    # #featureScaling.stdNormalization(mlModel)
    # train.gradientDescent(mlModel)
    # plots.plotCost(mlModel)
 
class Model():

    def __init__(self, model, trainDataFileName, dataFormat, testDataFileName=None, netArchitecture=None, learningRate=0.0001, regularization=0.1, iterations=10, randInitRange=0.1):
        self.model = model
        self.trainDataFileName = trainDataFileName
        self.testDataFileName = testDataFileName
        self.netArchitecture = netArchitecture
        self.learningRate = learningRate
        self.regularization = regularization
        self.iterations = iterations
        self.randInitRange = randInitRange


def main():
    
    # Can choose from linearRegression, logisticRegression and neuralNet
    model = "linearRegression"
    
    # Training data
    # The files containing the training data should be placed in the data folder and the 
    # names of the files should be specified below. Features are are arranged so each row is a
    # training example and each feature is a column. Labels should be the last column.
    trainDataFileName = "linearTrain.csv"
    testDataFileName = None #"linearTest.csv"

    # Format of the training data
    # Valid options are csv or numpy array
    fileFormat = "csv"

    # Hyperparameters
    learningRate = 0.0001
    regularization = 0.1
    randInitRange = 0.1
    
    # Number of iterations to run the learning algorithm for
    iterations = 10

    # If the model used is a neural net you can specify the number of nodes in each layer using a list.
    # For example a neural network with an input layer of size 10, hidden layer 12 and output layer of 8
    # would use [10, 12, 8]. The input layer must be the same size as the number of features (columns) in
    # the training data. If you are not using a neural net the list can be left empty.
    netArchitechture = []

    mlModel = Model(model, 
                    trainDataFileName, 
                    fileFormat, 
                    netArchitecture=netArchitechture,
                    learningRate=learningRate, 
                    regularization=regularization, 
                    iterations=iterations,
                    randInitRange=randInitRange)

    run(mlModel)

# TODO Feature Scaling options
# TODO Stochastic vs Gradient 
# TODO Randomly Shuffle Data
# TODO Split training set into test, cv and train (% split?)
# TODO Add more data formats



if __name__ == "__main__":
    main()