import random

from collections import deque
from typing import final

import numpy as np
import torch

from snake import SnakeGame
from utils.Snake_Statics import HIDDEN_SIZE, INPUT_SIZE, OUTPUT_SIZE, 
import utils.AIUtils as AIU
from models.torchModel import *

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class QAgent :
    def __init__(self) :
        self.n = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game) :
        return AIU.get_inputs(INPUT_SIZE, game)

    def remember(self, state, action, reward, next_state, done) :
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) :
        if len(self.memory) > BATCH_SIZE :
            mini_Sample = random.sample(self.memory, BATCH_SIZE)
        else :
            mini_Sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_Sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done) :
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state) :
        self.epsilon = 80 - self.n
        final_action = [0, 0, 0, 0]

        if random.randint(0,200) < self.epsilon :
            move = random.randint(0,3)
            final_action[move] = 1
        else :
            state0 = torch.tensor(state, dtype=torch.float)
            pred = self.model(state0)
            move = torch.argmax(pred).item()
            final_action[move] = 1

        return final_action

def train() :
