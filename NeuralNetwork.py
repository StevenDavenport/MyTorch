import numpy as np

from Layer import Layer
from Loss import LossFunction
from Activation import ActivationFunction
from Optimizer import Optimizer

class NeuralNetwork:
    def __init__(self, layers: list, loss: LossFunction, optim: Optimizer) -> None:
        self.layers = layers
        self.loss = loss
        self.optim = optim
    
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def train():
        pass



def main():
    from Activation import ReLU, Sigmoid, Softmax
    from Layer import FullyConnected
    from Loss import MSE
    from Optimizer import SGD

    # xor input
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    
    # Test input
    test_input = np.array([[0, 0]])

    # Neural Network
    net = NeuralNetwork([
        FullyConnected(2, 3, ReLU()),
        FullyConnected(3, 1, Sigmoid())
    ])

    # Train
    net.train(X, y, epochs=1000, learning_rate=0.01)


if __name__ == '__main__':
    main()