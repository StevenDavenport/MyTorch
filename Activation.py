import numpy as np

class Activation:
    '''
    Activation function base class.
    Usage:
        >>> Inherited by other activation functions.
        >>> Actication and derivative functions.
        >>> Activation__call__(inputs)    -> np.ndarray
        >>> Activation.derivative(inputs) -> np.ndarray
    '''
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        '''
        Activation function.
        Parameters:
            inputs: np.ndarray
        Returns:
            np.ndarray
        '''
        raise NotImplementedError

    def derivative(self, inputs: np.ndarray) -> np.ndarray:
        '''
        Derivative of activation function.
        Parameters:
            inputs: np.ndarray
        Returns:
            np.ndarray
        '''
        raise NotImplementedError

class ReLU(Activation):
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return np.maximum(0, inputs)

    def derivative(self, inputs: np.ndarray) -> np.ndarray:
        return np.where(inputs > 0, 1, 0)

class Sigmoid(Activation):
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-inputs))

    def derivative(self, inputs: np.ndarray) -> np.ndarray:
        return self(inputs) * (1 - self(inputs))

class Softmax(Activation):
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_values / np.sum(np.exp(inputs), axis=1, keepdims=True)
        
    def derivative(self, inputs: np.ndarray) -> np.ndarray:
        return self(inputs) * (1 - self(inputs))