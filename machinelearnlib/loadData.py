from numpy import genfromtxt

def loadData(mlModel):
    """
    Load in data from the dat folder using the file names specified in the mlModel
    """
    trainFileName = './data/' + mlModel['trainDataFileName']
    testFileName = './data/' + mlModel['testDataFileName']
    trainData = genfromtxt(trainFileName, delimiter=',')
    testData = genfromtxt(testFileName, delimiter=',')


    mlModel['features'] = trainData[:, :-1]
    mlModel['labels'] = trainData[:, -1]

    mlModel['testFeatures'] = testData[:, :-1]
    mlModel['testLabels'] = testData[:, -1]


