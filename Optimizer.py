import numpy as np

from Layer import Layer

class Optimizer:
    '''
    Optimizer base class.
    Inherited by other optimizer classes.
        >>> update_params(layer) == Update parameters of layer.
            >>> In future __call__(layers)
    '''
    def update_params(self, layer: Layer):
        '''
        Update parameters of layer.
        Parameters:
            layer: Layer
        Returns:
            None
        '''
        raise NotImplementedError

class SGD(Optimizer):
    '''
    Optimizer class. 
    Stochastic Gradient Descent.
    Can be defined with/without decay & momentum.
    '''
    def __init__(self, learning_rate: float = 1., decay: float = 0., momentum: float = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self): 
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer): 
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * \
                            layer.dweights
            bias_updates = -self.current_learning_rate * \
                            layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self): 
        self.iterations += 1
        

