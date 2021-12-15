class SGD:
    '''
    Stochastic gradient descent optimizer.
    '''
    def __init__(self, learning_rate=0.01):
        '''
        Initialize the optimizer.
        '''
        self.learning_rate = learning_rate
        
    def update(self, layers):
        '''
        Update the layer parameters.
        '''
        for layer in layers:
            layer.weights -= self.learning_rate * layer.grad_weights
            layer.bias -= self.learning_rate * layer.grad_bias
