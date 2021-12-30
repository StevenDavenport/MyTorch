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

class Softmax(ActivationFunction):
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

    def deriv(self, dvalues: np.ndarray) -> np.ndarray:
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T) 
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)