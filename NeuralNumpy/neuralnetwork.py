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
            FullyConnected(input_shape=(2,), output_shape=(1,), activation=Sigmoid())],
        loss=MSE(),
        optimizer=SGD())

    # Create the XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    # Train the network
    for i in range(1000):
        idx = np.random.randint(0, 4)
        input = X[idx]
        input = np.reshape(input, (1, 2))
        net.forward(input)
        net.backward(Y[idx])
        net.update()
    

    # Test the network
    print(net.forward(np.array([0, 0])))
    print(net.forward(np.array([0, 1])))
    print(net.forward(np.array([1, 0])))
    print(net.forward(np.array([1, 1])))



# function that uses pytorch to make and train a neural network
def torch_nn():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.autograd import Variable

    # create the neural network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 1)
            self.act1 = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.act1(x)
            return x
        
    # create the XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    # train the network
    net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    for epoch in range(1000):
        inputs = Variable(torch.from_numpy(X))
        targets = Variable(torch.from_numpy(Y))

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # test the network
    print(net(Variable(torch.from_numpy(np.array([0, 0])))))
    print(net(Variable(torch.from_numpy(np.array([0, 1])))))
    print(net(Variable(torch.from_numpy(np.array([1, 0])))))
    print(net(Variable(torch.from_numpy(np.array([1, 1])))))


if __name__ == '__main__':
    test()
    #torch_nn()