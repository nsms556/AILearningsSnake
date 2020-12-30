import numpy as np

class SnakeNet() :
    def __init__(self, hidden = 10) :
        self.fitness = 0
        self.hidden = hidden

        self.w1 = np.random.randn(24, hidden)
        self.w2 = np.random.randn(hidden, hidden)
        self.w3 = np.random.randn(hidden, 4)

    def forward(self, inputs) :
        net = np.matmul(inputs, self.w1)
        net = self.relu(net)
        net = np.matmul(net, self.w2)
        net = self.relu(net)
        net = np.matmul(net, self.w3)
        net = self.softmax(net)

        return net

    def relu(self, x) :
        return x * (x >= 0)

    def softmax(self, x) :
        return np.exp(x) / np.sum(np.exp(x), axis = 0)

    def leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01)