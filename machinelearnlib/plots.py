import matplotlib.pyplot as plt
import numpy as np

def plotTrainData(mlModel):
    """
    This function is only recomended for data sets with a small number of features. Data sets with multiple
    features are difficult to visualize on a 2d plot. If a regression model is used this function will create
    a plot for each feature against the label. If a classification model is used the each feature will be plotted
    against each other with a different color marker for each class.
    """

    # TODO classification plot

    features = mlModel['features']
    labels = mlModel['labels']
    numFeatures = features.shape[1]

    if mlModel["model"] == "linearRegression":
        for i in range(numFeatures):
            plt.scatter(features[:, i], labels)
            plt.title('Training Data')
            plt.xlabel('feature ' + str(i))
            plt.ylabel('labels')
            plt.show()    
        
def plotPrediction(mlModel):
    
    features = mlModel['features']
    labels = mlModel['labels']
    numFeatures = features.shape[1]

    if mlModel["model"] == "linearRegression":
        for i in range(numFeatures):
            # Plot training data
            plt.scatter(features[:, i], labels)

            # Plot prediction
            featuresUni = np.array(np.arange(100)).reshape(100, 1)
            import models.linearRegression as linearRegression
            predictions = linearRegression.predict(featuresUni, mlModel['weights'])
            plt.plot(featuresUni, predictions, 'r')


            plt.title('Training Data')
            plt.xlabel('feature ' + str(i))
            plt.ylabel('labels')
            plt.show()   
