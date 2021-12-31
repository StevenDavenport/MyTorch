import numpy as np

class LossFunction:
    '''
    Loss function base class.
    Usage:
        >>> Inherited by other loss functions.
        >>> LossFunction__call__(y, y_hat) -> float
    '''
    def __call__(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        '''
        Loss function.
        Parameters:
            y: np.ndarray
            y_hat: np.ndarray
        Returns:
            float
        '''
        raise NotImplementedError

    def deriv(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        '''
        Derivative of loss function.
        Parameters:
            y: np.ndarray
            y_hat: np.ndarray
        Returns:
            float
        '''
        raise NotImplementedError

class CategoricalCrossentropy(LossFunction): 
    def __call__(self, y_hat, y):
        sample_losses = self.forward(y_hat, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def forward(self, y_pred, y_true): 
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # This is 'backward' from Activation_Softmax_Loss_CategoricalCrossentropy in book.py
    def deriv(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
        return self.dinputs

   # This is backward from Loss_CategoricalCrossentropy in book.py 
    '''def deriv(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
        return self.dinputs'''
