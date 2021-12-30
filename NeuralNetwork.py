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

    def backward(self):
        for layer in reversed(self.layers):
            self.dinputs = layer.backward(self.dinputs)
            self.optim.update_params(layer)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 1) -> None:
        for epoch in range(epochs):
            y_hat = self(X)
            loss = self.loss(y_hat, y)
            pred = np.argmax(y_hat, axis=1)
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
            accuracy = np.mean(pred == y)
            if not epoch % 100:
                print(f'Epoch: {epoch}, Accuracy: {accuracy:.3f}, Loss: {loss:.3f}, Learning rate: {self.optim.current_learning_rate:.3f}')
            self.dinputs = self.loss.deriv(y_hat, y)
            self.optim.pre_update_params()
            self.backward()
            self.optim.post_update_params()


def test():
    from NeuralNetwork import NeuralNetwork
    from Layer import FullyConnected
    from Activation import ReLU, Softmax
    from Loss import CategoricalCrossentropy
    from Optimizer import SGD

    import nnfs
    from nnfs.datasets import spiral_data
    nnfs.init()

    X, y = spiral_data(samples=100, classes=3)

    net = NeuralNetwork(
        [
            FullyConnected(2, 64, ReLU()),
            FullyConnected(64, 3, Softmax())
        ],
        loss=CategoricalCrossentropy(),
        optim=SGD(learning_rate=1., decay=1e-3, momentum=0.9)
    )

    net.train(X, y, epochs=10001)

if __name__ == '__main__':
    test()
