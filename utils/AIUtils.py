import numpy as np

import utils.utils as U
from utils.Snake_Statics import DETECT_DIRS

def get_inputs(inputSize, game) :
    baseInput = np.zeros(shape = inputSize)
    
    for i, direction in enumerate(DETECT_DIRS) :
        baseInput[i*3:i*3+3] = detection(game.player.posList, game.item.position, direction)
    
    return baseInput

def detection(bodyList, itemP, direction) :
    sensingPoint = bodyList[0].copy()
    distance = 0
    detectItem = False
    detectBody = False

    sensingPoint += direction
    distance += 1
    
    detect = np.zeros(3)
    while not U.wallCollide(sensingPoint) :
        if not detectItem and U.itemCollide(sensingPoint, itemP) :
            detectItem = True
            detect[0] = 1

        if not detectBody and U.bodyCollide(sensingPoint, bodyList) :
            detectBody = True
            detect[1] = 1

        sensingPoint += direction
        distance += 1
        
    detect[2] = 1 / distance
    
    return detect