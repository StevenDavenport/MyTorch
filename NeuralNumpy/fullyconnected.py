import numpy as np
from layer import Layer

class FullyConnected(Layer):
    '''
    A fully connected layer.
    '''
    def __init__(self, input_shape, output_shape, activation):
        '''
        Initialize the layer.
        '''
        super(FullyConnected, self).__init__(input_shape, output_shape)
        self.weights = np.random.randn(input_shape[0], output_shape[0])
        self.bias = np.random.randn(output_shape[0])
        self.grad_weights = np.zeros(self.weights.shape)
        self.grad_bias = np.zeros(self.bias.shape)
        self.activation = activation
        
    def forward(self, input):
        '''
        Perform the forward pass.
        '''
        self.input = input
        self.output = self.activation.activate(np.dot(input, self.weights) + self.bias)
        return self.output

    def backward(self, grad_output):
        '''
        Perform the backward pass.
        '''
        m = grad_output.size
        self.delta = grad_output * self.activation.derivative(self.input)
        self.grad_weights = 1 / m * np.dot(self.delta, self.input.T)
        self.grad_bias = 1 / m * np.sum(self.delta, axis=1)
        return self.delta
    
    def update(self, learning_rate):
        '''
        Update the parameters of the layer.
        '''
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias