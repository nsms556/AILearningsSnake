import numpy as np
from copy import deepcopy
from random import choice, randint, uniform

from Snake_Statics import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE

class SnakeNet() :
    def __init__(self, hidden = HIDDEN_SIZE, pre_weight = None) :
        self.fitness = 0
        self.hidden = hidden

        if pre_weight is None :
            self.w1 = np.random.randn(INPUT_SIZE, hidden)
            self.w2 = np.random.randn(hidden, hidden)
            self.w3 = np.random.randn(hidden, OUTPUT_SIZE)
            self.weights = np.array([self.w1, self.w2, self.w3], dtype=object)
        else :
            self.weights = np.load(pre_weight, allow_pickle=True)
            self.w1, self.w2, self.w3 = self.weights

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

    def sigmoid(self, x) :
        return 1 / (1 + np.exp(-x))

    def crossover(self, crossModel) :
        childModel = deepcopy(self)

        cut = randint(0, childModel.w1.shape[1])
        childModel.w1[0, cut:] = crossModel.w1[0, cut:]

        cut = randint(0, childModel.w2.shape[1])
        childModel.w2[0, cut:] = crossModel.w2[0, cut:]

        cut = randint(0, childModel.w3.shape[1])
        childModel.w3[0, cut:] = crossModel.w3[0, cut:]

        return childModel

    def mutate(self, bestModel, rate) :
        childModel = deepcopy(self)

        mutation = np.random.random_sample(childModel.w1.shape) < rate
        uniform_mutation = np.random.uniform(size = childModel.w1.shape)
        childModel.w1[mutation] += uniform_mutation[mutation] * (bestModel.w1[mutation] - childModel.w1[mutation])

        mutation = np.random.random_sample(childModel.w2.shape) < rate
        uniform_mutation = np.random.uniform(size = childModel.w2.shape)
        childModel.w2[mutation] += uniform_mutation[mutation] * (bestModel.w2[mutation] - childModel.w2[mutation])

        mutation = np.random.random_sample(childModel.w3.shape) < rate
        uniform_mutation = np.random.uniform(size = childModel.w3.shape)
        childModel.w3[mutation] += uniform_mutation[mutation] * (bestModel.w3[mutation] - childModel.w3[mutation])

        return childModel