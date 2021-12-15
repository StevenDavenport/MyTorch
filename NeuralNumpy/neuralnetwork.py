import numpy as np

class NeuralNetwork:
    '''
    A neural network.
    '''
    def __init__(self, layers):
        '''
        Initialize the network.
        '''
        self.layers = layers
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
    
    def backward(self, grad_output):
        '''
        Perform the backward pass.
        '''
        self.grad_output = grad_output
        for layer in reversed(self.layers):
            grad_output = layer.backward(self.input, grad_output)
        return self.grad_output
    
    def update(self, learning_rate):
        '''
        Update the parameters of the network.
        '''
        for layer in self.layers:
            layer.update(learning_rate)


def test():
    from fullyconnected import FullyConnected
    from activation import ReLU, Sigmoid, Softmax

    '''
    Test the neural network class by 
        - creating a neural netwwork 
        - Testing it on the XOR problem
    '''
    # Create the network to solve the XOR problem
    net = NeuralNetwork([
        FullyConnected((2,), (1,), Sigmoid())
    ])

    # Create the XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    # Train the network
    for i in range(10000):
        # Forward pass
        output = net.forward(X)

        # Compute the loss
        loss = np.mean(np.square(output - Y))
        #print('Loss : ', loss)

        # Backward pass
        net.backward(Y)

        # Update the parameters
        net.update(0.01)
    
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
    #test()
    torch_nn()