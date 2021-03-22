import random

from collections import deque

import numpy as np
import torch

from snake import SnakeGame
from models.torchModel import *

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class QAgent :
    def __init__(self) :
        
