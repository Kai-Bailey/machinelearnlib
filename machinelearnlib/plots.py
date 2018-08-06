import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def plotTrainDataLinearRegression(mlModel):
    """
    This function is only recommended for data sets with a small number of features. Data sets with multiple
    features are difficult to visualize on a 2d plot. This will create a plot of each feature against the labels.
    """ 

    features = mlModel.features
    labels = mlModel.labels
    numFeatures = features.shape[1]

    for i in range(numFeatures):
        plt.scatter(features[:, i], labels)
        plt.title('Training Data')
        plt.xlabel('feature ' + str(i))
        plt.ylabel('labels')
        plt.show()

    return mlModel

def plotTrainDataLogisticRegression(mlModel, feature1, feature2):
    """
    This function is only recommended for data sets with a small number of features. Data sets with multiple
    features are difficult to visualize on a 2d plot. This plot the two features against each other with different
    markers for each label. Feature1 and feature2 are the column numbers of the features to plot. 
    """

    features = mlModel.features
    labels = mlModel.labels

    uniqueLabels = {}
    colors = ['k', 'y', 'c', 'm', 'g', 'r', 'b']
    for label in labels:
        if label not in uniqueLabels:
            uniqueLabels[label] = colors.pop()

    
    for feat1, feat2, label in zip(features[:, feature1], features[:, feature2], labels):
        plt.scatter(feat1, feat2, c=uniqueLabels[label])

    # Create a legend mapping color of the marker to the label of the data
    patches = []
    for label in uniqueLabels:
        patches.append(mpatches.Patch(color=uniqueLabels[label], label=str(label)))
    
    plt.legend(handles=patches)


    plt.title('Feature ' + str(feature1) + ' VS ' + 'Feature ' + str(feature2))
    plt.xlabel('feature ' + str(feature1))
    plt.ylabel('feature ' + str(feature2))
    plt.show()

    return mlModel

def plotPrediction(mlModel):
    
    features = mlModel.features
    labels = mlModel.labels
    numFeatures = features.shape[1]

    if mlModel.model == "linearRegression":
        for i in range(numFeatures):
            # Plot training data
            plt.scatter(features[:, i], labels)

            # Plot prediction
            featuresUni = np.array(np.arange(100)).reshape(100, 1)
            predictions = mlModel.predict(featuresUni, mlModel.weights)
            plt.plot(featuresUni, predictions, 'r')


            plt.title('Training Data')
            plt.xlabel('feature ' + str(i))
            plt.ylabel('labels')
            plt.show()   

    return mlModel

def plotCost(mlModel):
    """
    Plots the cost for each iteration. Note that a training algorithm must be run before this plot.
    """

    plt.plot(mlModel.cost)
    plt.title('Cost vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

    return mlModel