import numpy as np
from random import randint

HM_FPS = 15
NN_FPS = 60

SCREEN_SIZE = 20
PIXEL_SIZE = 20
LINE_WIDTH = 1

DIRECTIONS = np.array([
    (-1, 0),    # Up
    (0, 1),     # Right
    (0, -1),    # Left
    (1, 0)      # Down
])

DETECT_DIRS = np.array([
    (-1, 0), (-1, -1), (0, -1), (1, -1),
    (1, 0), (1, 1), (0, 1), (-1, 1)
])

UP = 0
DOWN = 3
LEFT = 2
RIGHT = 1
MOVE_LIMIT = 3

MOVE_NN_DIR = [UP, RIGHT, LEFT, DOWN]

WHITE = (255,255,255)
GREEN = (  0,255,  0)
BLACK = (  0,  0,  0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

DISPLAY_WIDTH = SCREEN_SIZE * PIXEL_SIZE
DISPLAY_HEIGHT = SCREEN_SIZE * PIXEL_SIZE

def randint_20() :
    return randint(0, 19)

class BreakException(Exception) :
    pass

N_POPULATION = 50
N_BEST = 5
N_CHILDREN = 5
MUTATION_RATE = 0.04
LIFE_LEFT = NN_FPS * 5

SEE_VISION = False
SELECT_FIT = 2

INPUT_SIZE = 24
HIDDEN_SIZE = 10
OUTPUT_SIZE = 4

