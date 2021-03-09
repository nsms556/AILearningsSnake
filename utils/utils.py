import numpy as np

from utils.Snake_Statics import SCREEN_SIZE

def calcFitness(lifeTime, score) :
    fitness = (lifeTime) + ((2**score) + (score**2.1)*500) - (((.25 * lifeTime)**1.3) * (score**1.2))
    
    return round(fitness, 1)

def wallCollide(point) :
    if point[0] >= SCREEN_SIZE or point[0] < 0 or point[1] >= SCREEN_SIZE or point[1] < 0 :
        return True
    
    return False

def bodyCollide(point, body) :
    if point.tolist() in body[1:].tolist() :
        return True

    return False

def itemCollide(point, itemP) :
    if all(point == itemP) :
        return True

    return False

if __name__ == '__main__' :
    print('TEST utils.py')
    score = 1
    lifeTime = 1
    positionList = np.array([(2,1), (1,1), (0,1)])
    itemPos = np.array([2,1])
    testPos1 = np.array([1,1])
    testPos2 = np.array([5, 20])

    print('Fitness : {}'.format(calcFitness(lifeTime, score)))
    print('Wall(1,1) : {}'.format(wallCollide(testPos1)))
    print('Wall(5,20) : {}'.format(wallCollide(testPos2)))
    print('Body(1,1) :{}'.format(bodyCollide(testPos1, positionList)))
    print('Item(2,1) : {}'.format(itemCollide(positionList[0], itemPos)))
