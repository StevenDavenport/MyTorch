import numpy as np
from Layer import Layer

class NeuralNetwork:
    def __init__(self, layers: list) -> None:
        self.layers = layers
    
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs



if __name__ == '__main__':
    from Activation import ReLU, Sigmoid, Softmax
    from Layer import FullyConnected
    import matplotlib.pyplot as plt

    def spiral_data(points, classes):
        X = np.zeros((points*classes, 2))
        y = np.zeros(points*classes, dtype='uint8')
        for class_number in range(classes):
            ix = range(points*class_number, points*(class_number+1))
            r = np.linspace(0.0, 1, points)  # radius
            t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
            X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
            y[ix] = class_number
        return X, y

    X, y = spiral_data(100, 3)

    net = NeuralNetwork([
        FullyConnected(2, 3, ReLU()),
        FullyConnected(3, 3, Softmax())
    ])

    print(net(X))
