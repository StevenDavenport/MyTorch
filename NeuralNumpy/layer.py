import numpy as np
from activation import ReLU, Sigmoid, Softmax

class Layer:
    '''
    A parent class for all layers.
    '''
    def __init__(self, input_shape, output_shape):
        '''
        Initialize the layer.
        '''
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        '''
        Perform the forward pass.
        '''
        raise NotImplementedError

    def backward(self, input, grad_output):
        '''
        Perform the backward pass.
        '''
        raise NotImplementedError

    def update(self, learning_rate):
        '''
        Update the parameters of the layer.
        '''
        raise NotImplementedError
    