import numpy as np
from Activation import ActivationFunction

class Layer:
    '''
    Layer base class.
    Usage:
        >>> Inherited by other layer classes.
        >>> Layer__call__(inputs) == Forward Pass -> np.ndarray
    '''

    def __init__(self, n_features: int, n_outputs: int, activation: ActivationFunction):
        '''
        Layer initialization.
        Parameters:
            n_features: int
            n_outputs: int
            activation: Activation
        Returns:
            None
        '''
        raise NotImplementedError

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        '''
        Layer function.
        Parameters:
            inputs: np.ndarray
        Returns:
            np.ndarray
        '''
        raise NotImplementedError
    
    def backward(self, outputs: np.ndarray) -> np.ndarray:
        '''
        Layer backward function.
        Calculate the gradient of the layer.
        Parameters:
            inputs: np.ndarray
        Returns:
            np.ndarray
        '''
        raise NotImplementedError

class FullyConnected(Layer):
    def __init__(self, n_features: int, n_outputs: int, activation: ActivationFunction) -> None:
        self.weights = 0.10 * np.random.randn(n_features, n_outputs)
        self.biases = np.zeros((1, n_outputs))
        self.activation = activation

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        self.output = self.activation(np.dot(inputs, self.weights) + self.biases)
        return self.output

    def backward():
        pass

