import buildModel
import loadData
import train
import plots
import initializeWeights

title = """
                  __   _          __                  ___ __ 
  __ _  ___ _____/ /  (_)__  ___ / /__ ___ ________  / (_) / 
 /  ' \/ _ `/ __/ _ \/ / _ \/ -_) / -_) _ `/ __/ _ \/ / / _ \\
/_/_/_/\_,_/\__/_//_/_/_//_/\__/_/\__/\_,_/_/ /_//_/_/_/_.__/

"""



def start(mlModel):
    print(title)
    loadData.loadData(mlModel)
    plots.plotTrainData(mlModel)
    initializeWeights.initializeWeights(mlModel)
    train.gradientDescent(mlModel)
 


def main():
    #mlModel = buildModel.buildModel()
    
    mlModel = {
    # Can choose from linearRegression, logisticRegression and neuralNet
    "model":"linearRegression",
    
    # Training data
    # The files containing the training data should be placed in the data folder and the 
    # names of the files should be specified below. Features are are arranged so each row is a
    # training example and each feature is a column. Labels should be the last column.
    "trainDataFileName":"train.csv",
    "testDataFileName":"test.csv",

    # Format of the training data
    # Valid options are csv
    "format":"csv",

    # Hyperparameters
    "learningRate":0.0001,
    "regularization":0.1,

    # If the model used is a neural net you can specify the number of nodes in each layer using a list.
    # For example a neural network with an input layer of size 10, hidden layer 12 and output layer of 8
    # would use [10, 12, 8]. The input layer must be the same size as the number of features (columns) in
    # the training data. If you are not using a neural net the list can be left empty.
    "netArchitechture":[]
    }
    
    start(mlModel)

if __name__ == "__main__":
    main()