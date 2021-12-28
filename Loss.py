import numpy as np

class LossFunction:
    '''
    Loss function base class.
    Usage:
        >>> Inherited by other loss functions.
        >>> LossFunction__call__(y, y_hat) -> float
    '''
    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        '''
        Loss function.
        Parameters:
            y: np.ndarray
            y_hat: np.ndarray
        Returns:
            float
        '''
        raise NotImplementedError

    def derivative(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        '''
        Derivative of loss function.
        Parameters:
            y: np.ndarray
            y_hat: np.ndarray
        Returns:
            float
        '''
        raise NotImplementedError

class MSE(LossFunction):
    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        return np.mean((y - y_hat) ** 2)

    def derivative(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        return 2 * (y - y_hat)

class CrossEntropy(LossFunction):
    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        return -np.mean(y * np.log(y_hat))

    def derivative(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        return -y / y_hat

class SoftmaxCrossEntropy(LossFunction):
    def __call__(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        return -np.mean(y * np.log(y_hat))

    def derivative(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        return -y / y_hat
