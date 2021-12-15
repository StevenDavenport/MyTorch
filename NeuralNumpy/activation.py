import numpy as np

class Activation:
    def activate(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError

class ReLU(Activation):
    def activate(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return 1. * (x > 0)
    
class Sigmoid(Activation):
    def activate(self, x):
        return 1. / (1. + np.exp(-x))

    def derivative(self, x):
        return self.activate(x) * (1. - self.activate(x))

class Softmax(Activation):
    def activate(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def derivative(self, x):
        return self.activate(x) * (1. - self.activate(x))