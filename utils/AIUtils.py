import numpy as np

import utils.utils as U

def get_inputs(positions, itemP, inputSize, directions) :
    baseInput = np.zeros(shape = inputSize)
    
    for i, direction in enumerate(directions) :
        baseInput[i*3:i*3+3] = detection(positions, itemP, direction)
    
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