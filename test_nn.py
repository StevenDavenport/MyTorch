import numpy as np
from numpy.random.mtrand import f

# dataset - XOR problem
train_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_Y = np.array([[0], [1], [1], [0]])

def init_params():
    weights = np.random.randn(2, 1)
    bias = np.random.randn(1)
    return weights, bias

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def softmax_deriv(x):
    return softmax(x) * (1 - softmax(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def forward_prop(weights, bias, x):
    z = np.dot(x, weights) + bias
    a = sigmoid(z)
    return z, a

def back_prop(weights, bias, z, x, y):
    m = x.size
    dz = x - y
    dw = 1. / m * dz.dot(x.T)
    db = 1. / m * np.sum(dz)
    return dw, db

def update_params(weights, bias, dw, db, alpha):
    weights -= alpha * dw
    bias -= alpha * db
    return weights, bias

def get_predictions(a):
    return 1 if a >= 0.50 else 0

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def predict(weights, bias, x):
    z, a = forward_prop(weights, bias, x)
    return get_predictions(a)

def gradient_descent(X, Y, n, alpha):
    weights, bias = init_params()
    for i in range(n):
        r = np.random.randint(X.shape[0])
        z, a = forward_prop(weights, bias, X[r])
        dw, db = back_prop(weights, bias, z, a, Y[r])
        weights, bias = update_params(weights, bias, dw, db, alpha)
        if (i % 10 == 0):
            print(f'Iteration: {i}, Accuracy: {get_accuracy(get_predictions(a), Y[r])}')
    return weights, bias

def main():
    weights, bias = gradient_descent(train_X, train_Y, 1000, 0.01)
    print()
    for i in range(4):
        print(f'Input: {train_X[i]}, Prediction: {predict(weights, bias, train_X[i])}, Actual: {train_Y[i][0]}')

if __name__ == '__main__':
    main()