import numpy as np

def sigmoid(arr):
    """
    Passes each element of the numpy array arr through the sigmiod function elementwise.

    :param arr: numpy array
    :return: arr passed through sigmoid function elementwise
    """

    return 1/(1 + np.exp(-1 * arr))

def sigmoid_gradient(arr):
    """
    Compute the gradient of the sigmoid function for each element in arr,

    :param arr: numpy array
    :return: arr passed through the gradient of the sigmoid function elementwise
    """

    return sigmoid(arr) * (1-sigmoid(arr))
