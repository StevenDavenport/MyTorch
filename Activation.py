import numpy as np

class ActivationFunction:
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

    def derivative(self, dvalues: np.ndarray) -> np.ndarray:
        '''
        Derivative of activation function.
        Parameters:
            dvalues: np.ndarray
        Returns:
            np.ndarray
        '''
        raise NotImplementedError

class ReLU(ActivationFunction):
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def deriv(self, dvalues: np.ndarray) -> np.ndarray:
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs