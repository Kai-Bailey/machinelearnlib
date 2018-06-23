import numpy as np

def sigmoid(arr):
    """
    Passes each element of the numpy array arr through the sigmiod function elementwise.

    Param: arr - numpy array
    Return: arr passed through sigmoid function elementwise
    """

    return np.exp(arr)/(np.exp(arr) + 1)
