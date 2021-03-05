import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from Snake_Statics import *

class SnakeNet(nn.Module) :
    def __init__(self) :
        super(SnakeNet, self).__init__()

        self.linear1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.linear2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x) :
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x

    def save(self, fileName = 'model.pth') :
        filePath = './weights'

        fullPath = os.path.join(filePath, fileName)
        torch.save(self.state_dict(), fullPath)