''' 
This file uses comments to explain a backpropagation algorithm.
'''

# Import the numpy package
import numpy as np

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network class
class NeuralNetwork:
    # Initialize the neural network
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1], 4)
        self.weights2   = np.random.rand(4, 1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    # Calculate the output of the neural network
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    # Backpropagate the error
    def backprop(self):
        # Calculate the error
        self.error = self.y - self.output
        # Multiply the error by the derivative of the sigmoid function
        self.delta = self.error * sigmoid_derivative(self.output)
        # Calculate the gradient
        self.grad2 = np.dot(self.layer1.T, self.delta)
        # Calculate the error
        self.error = self.delta.dot(self.weights2.T)
        # Multiply the error by the derivative of the sigmoid function
        self.delta = self.error * sigmoid_derivative(self.layer1)
        # Calculate the gradient
        self.grad1 = np.dot(self.input.T, self.delta)

    # Update the weights
    def update(self, learning_rate):
        self.weights1 += learning_rate * self.grad1
        self.weights2 += learning_rate * self.grad2

def main():
    # Create XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Create a neural network
    nn = NeuralNetwork(X, y)

    # Train the neural network
    for i in range(10000):
        nn.feedforward()
        nn.backprop()
        nn.update(0.05)

    # Test the neural network
    print("Predictions:")
    # Make test predictions
    print(nn.feedforward(np.array([0, 0])))
    print(nn.feedforward(np.array([0, 1])))
    print(nn.feedforward(np.array([1, 0])))
    print(nn.feedforward(np.array([1, 1])))

if __name__ == "__main__":
    main()