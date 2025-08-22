import numpy as np

def identity(x):
    return x

def step(x):
    return np.array(x > 0, dtype=int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x >= 0] = 1
    return grad

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        x = np.exp(x) / np.sum(np.exp(x), axis=0)
        return x.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def mean_squared_error(x, t):
    return 0.5 * np.sum((x - t) ** 2)

def cross_entropy_error(x, t):
    if x.ndim == 1:
        x = x.reshape(1, x.size)
        t = t.reshape(1, t.size)

    if x.size == t.size:
        t = t.argmax(axis=1)

    batch_size = x.shape[0]

    return -np.sum(np.log(x[np.arange(batch_size), t] + 1e-7)) / batch_size

def softmax_loss(x, t):
    y = softmax(x)
    return cross_entropy_error(y, t)