import math
import random

from collections import namedtuple
from itertools import count

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from utils.Snake_Statics import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object) :
    def __init__(self, capacity) :
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args) :
        if len(self.memory) < self.capacity :
            self.memory.append(None)
        
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1 ) % self.capacity

    def sample(self, batchSize) :
        return random.sample(self.memory, batchSize)

    def __len__(self) :
        return len(self.memory)

class DQN(nn.Module) :
    def __init__(self, h, w, outputs) :
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 5, stride=2)
        