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
        self.grad_input = np.dot(grad_output, self.weights.T) * self.activation.derivative(self.input)
        self.grad_weights = np.dot(self.input.T, grad_output)
        self.grad_bias = np.sum(grad_output, axis=0)
        return self.grad_input
    
    def update(self, learning_rate):
        '''
        Update the parameters of the layer.
        '''
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias