import numpy as np

class MSE:
    '''
    Mean Squared Error
    '''
    def __call__(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

class CrossEntropy:
    '''
    Cross Entropy
    '''
    def __call__(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred))

class BinaryCrossEntropy:
    '''
    Binary Cross Entropy
    '''
    def __call__(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
