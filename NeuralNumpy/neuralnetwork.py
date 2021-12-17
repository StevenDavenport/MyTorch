import numpy as np

class NeuralNetwork:
    '''
    A neural network.
    '''
    def __init__(self, layers, loss, optimizer):
        '''
        Initialize the network.
        '''
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.grad_output = None
        
    def forward(self, input):
        '''
        Perform the forward pass.
        '''
        self.input = input
        for layer in self.layers:
            input = layer.forward(input)
        self.output = input
        return self.output
    
    def backward(self, y_true):
        '''
        Perform the backward pass.
        '''
        self.grad_output = self.loss(y_true, self.output)
        for layer in reversed(self.layers):
            self.grad_output = layer.backward(self.grad_output)
        return self.grad_output
    
    def update(self):
        '''
        Update the parameters of the network using the optimizer.
        '''
        self.optimizer.update(self.layers)


def test():
    from fullyconnected import FullyConnected
    from activation import ReLU, Sigmoid, Softmax
    from loss import MSE
    from optimizer import SGD
    '''
    Test the neural network class by 
        - creating a neural netwwork 
        - Testing it on the XOR problem
    '''
    # Create the network to solve the XOR problem
    net = NeuralNetwork([
            FullyConnected(input_shape=(2,), output_shape=(1,1), activation=Sigmoid())],
        loss=MSE(),
        optimizer=SGD())

    # Create the XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    # Outputs to make a mean for Accuracy
    outputs = []
    actual = []

    # Train the network
    for epoch in range(10000):
        idx = np.random.randint(0, 4)
        input = X[idx]
        input = np.reshape(input, (1, 2))
        net_out = net.forward(input)
        out = 1 if net_out > 0.5 else 0
        outputs.append(net_out)
        actual.append(Y[idx][0])
        net.backward(Y[idx][0])
        net.update()
        loss = net.loss(Y[idx][0], net_out)
        print(f'Epoch: {epoch}, Output: {out}, Actual: {Y[idx][0]}, Loss: {loss}')

    print('\nAccuracy of Training:', np.mean(outputs == actual))
    

    # Test the network
    print(1 if net.forward(np.array([0, 0])) > 0.5 else 0)
    print(1 if net.forward(np.array([0, 1])) > 0.5 else 0)
    print(1 if net.forward(np.array([1, 0])) > 0.5 else 0)
    print(1 if net.forward(np.array([1, 1])) > 0.5 else 0)


if __name__ == '__main__':
    test()
    #torch_nn()