import numpy as np

class Layer:
    def __init__(self, input_size: int, output_size: int, activation: str) -> None:
        self.weights = np.random.rand(input_size, output_size)
        print('Weights for layer: \n', self.weights, '\n')
        self.bias = np.random.rand(1, output_size)
        self.activation = Activation(activation)

class Activation:
    def __init__(self, activation: str) -> None:
        self.activation = activation

    def activate(self, data: np.array) -> np.array:
        if self.activation == "sigmoid":
            return self.sigmoid(data)
        elif self.activation == "relu":
            return self.relu(data)
        elif self.activation == "softmax":
            return self.softmax(data)
        else:
            raise Exception("Activation function not found: {}".format(self.activation))

    def inverse(self, X: np.array) -> np.array:
        if self.activation == "sigmoid":
            return self.inverse_sigmoid(X)
        elif self.activation == "relu":
            return self.inverse_relu(X)
        elif self.activation == "softmax":
            return self.inverse_softmax(X)
        else:
            raise Exception("Inverse - Activation function not found: {}".format(self.activation))

    def sigmoid(self, X: np.array) -> np.array:
        return 1 / (1 + np.exp(-X))

    def inverse_sigmoid(self, X: np.array) -> np.array:
        return np.log(X / (1 - X))

    def relu(self, X: np.array) -> np.array:
        return np.maximum(X, 0)
    
    def inverse_relu(self, X: np.array) -> np.array:
        return np.where(X > 0, 1, 0)

    def softmax(self, X: np.array) -> np.array:
        e_x = np.exp(X - np.max(X))
        return e_x / e_x.sum(axis=0)

    def ssoftmax(self, X: np.array) -> np.array:
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

    def inverse_softmax(self, X: np.array) -> np.array:
        return np.log(X)
    
class FullyConnected(Layer):
    def __init__(self, input_size: int, output_size: int, activation: str) -> None:
        super().__init__(input_size, output_size, activation)

class Error:
    def __init__(self, error: str) -> None:
        self.error = error

    def calculate(self, target: np.array, prediction: np.array) -> np.array:
        if self.error == "mse":
            return self.mse(target, prediction)
        elif self.error == "cross_entropy":
            return self.cross_entropy(target, prediction)
        else:
            raise Exception("Error function not found: {}".format(self.error))

    def mse(self, target: np.array, prediction: np.array) -> np.array:
        return np.mean((target - prediction) ** 2)

    def cross_entropy(self, target: np.array, prediction: np.array) -> np.array:
        return -np.mean(target * np.log(prediction))

class Optimizer:
    def __init__(self, optimizer: str, learning_rate: float) -> None:
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def update(self, layer: Layer, error: np.array) -> None:
        if self.optimizer == "sgd":
            self.sgd(layer, error)
        else:
            raise Exception("Optimizer function not found: {}".format(self.optimizer))

    def sgd(self, layer: Layer, error: np.array) -> None:
        layer.weights.data -= self.learning_rate * np.dot(layer.weights.data.T, error)
        layer.bias.data -= self.learning_rate * np.mean(error, axis=0)

class Network:
    def __init__(self, error: str, optimizer: str, learning_rate: float) -> None:
        self.num_layers = 0
        self.layers = []
        self.error = Error(error)
        self.optimizer = Optimizer(optimizer, learning_rate)

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)
        self.num_layers += 1

    def forward(self, X: np.array, batch_size=1) -> np.array:
        Y = list()
        if batch_size == 1:
            X = np.expand_dims(X, axis=1)
        for i in range(batch_size):
            for layer in self.layers:
                X[i] = np.dot(layer.weights, X[i]) + layer.bias
                X[i] = layer.activation.activate(X[i])
                print('Activation {} :\n{}'.format(i, X[i]))
                print('\n')
            Y.append(X[i])
        return np.array(Y)

    def backward(self, actual: np.array, prediction: np.array) -> None:
        error = self.error.calculate(actual, prediction)
        for i in reversed(range(self.num_layers)):
            self.optimizer.update(self.layers[i], error)


def main():
    # Load MNIST data using pytorch
    import torch
    import torchvision
    import torchvision.transforms as transforms

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training data
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    # Define the network
    net = Network(error='mse', optimizer='sgd', learning_rate=0.01)
    net.add_layer(FullyConnected(784, 256, 'relu'))
    net.add_layer(FullyConnected(256, 64, 'relu'))
    net.add_layer(FullyConnected(64, 10, 'softmax'))

    # Transform from pytorch tensor to numpy array and flatten and type float32
    def transform(tensor: torch.Tensor) -> np.array:
        X = tensor.detach().numpy().reshape(-1, 784).astype(np.float32)
        return X

    # Train the network
    for epoch in range(10):
        for data in trainloader:
            images, labels = data
            X = transform(images)
            Y = net.forward(X, batch_size=64)
            net.backward(actual=np.array(labels), prediction=np.array(Y))

    # Test the network
    correct = 0
    total = 0
    for batch in testloader:
        images, labels = data
        outputs = net.forward(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def test():
    net = Network(error='mse', optimizer='sgd', learning_rate=0.01)
    net.add_layer(FullyConnected(1, 4, 'relu'))
    net.add_layer(FullyConnected(4, 1, 'softmax'))

    X = np.array( 
                    [0.1,0.31,0.27,0.73]
                )

    X2 = np.array([ 
                    [0.1, 0.2], 
                    [0.1 ,0.2],
                    [0.3, 0.4],
                    [0.5, 0.6]
                ])

    print('Input : \n{}'.format(X))
    print('\n')

    Y = net.forward(X)

if __name__ == "__main__":
    #main()
    test()