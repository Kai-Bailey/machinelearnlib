import os
import processList
import loadModel
import loadData
import featureScaling
import plots
import initializeWeights
import train


class ModelError(Exception):
    """
    Exception wrapper for Model errors
    """
    pass

# A dictionary of process sequence for models loaded models
_models_sequence = {    "linearRegression" : [loadData.loadData,
                          plots.plotTrainDataLinearRegression,
                          initializeWeights.initializeWeights,
                          train.gradientDescent,
                          plots.plotPrediction,
                          plots.plotCost],

                        "logisticRegression" : [loadData.loadIrisData,
                          plots.plotTrainDataLinearRegression,
                          initializeWeights.initializeWeights,
                          train.gradientDescent,
                          plots.plotCost],

                        "neuralNet" : [loadData.loadDataMNIST,
                          initializeWeights.neuralNetInitializeWeights,
                          train.gradientDescentNeuralNetwork,
                          train.accuracyNeuralNetwork,
                          plots.plotCost]
                          }



def load_models(mlModel):
    """
    Calls the load function on all models in the model folder.

    :param mlModel: Class defining the users options for the ml model.
    :return mlModel: Class defining the users options for the ml model. 
    """

    model_path = os.path.abspath(os.path.dirname(__file__) + '/models/' + mlModel.model + '.py')

    if os.path.isfile(model_path):
        model = mlModel.model
    else:
        raise ModelError('Failed to load ' + model + ' : model has not been implemented yet')

    # Attempt to load model
    try:
        exec('import models.{}; models.{}.load(mlModel)'.format(model, model))
    except AttributeError:
        raise ModelError('Failed to load ' + model + ': model has no load function')
    except TypeError:
        raise ModelError('Failed to load ' + model + ': model returned invalid signature or was None')

    return mlModel

def load(model):
    """
    Loads the process list of a model
    :param model: The list of functions to execute for the specific model.
    """

    processList.append(_models_sequence[model])