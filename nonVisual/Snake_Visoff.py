import os, random
from datetime import timedelta, datetime
from time import sleep
import math

import numpy as np

from nonVisual.statics import *

class SnakeGame :
    def __init__(self, nn=None) :
        self.done = False
        self.nn = nn
        self.fitMethod = SELECT_FIT

    def play(self) :
        self.player = Snake()
        self.item = Item()
        
        lastCtrlTime = datetime.now()
        self.score = 0
        self.fitness = 0
        self.lastItemTime = 0
        self.lifeTime = 0
        self.lifeLeft = LIFE_LEFT

        while not self.done :
            if self.nn != None :
                inputs = self.get_inputs()
                outputs = self.nn.forward(inputs)
                result = np.argsort(outputs)

                for possible in result :
                    if self.player.direction + MOVE_NN_DIR[possible] != MOVE_LIMIT :
                        self.player.direction = MOVE_NN_DIR[possible]
                        lastCtrlTime = datetime.now()
                        break
            else :
                print('Net NotFound')
                self.done = True

            if all(self.player.posList[0] == self.item.position) :
                self.player.grow()
                self.score += 1
                self.lifeLeft = LIFE_LEFT + (self.score / 2 * NN_FPS)
                    
                self.item.newPosition()
                while self.item.position.tolist() in self.player.posList.tolist() :
                    self.item.newPosition()
            else :
                self.player.move()
                self.lifeLeft -= 1

            self.lifeTime += 1

            if self.player.posList[0].tolist() in self.player.posList[1:].tolist() :
                self.done = True
                
            if self.wallCollide(self.player.posList[0]) :
                self.done = True

            if self.lifeLeft <= 0 and self.nn is not None :
                self.done = True

            self.fitness = self.calcGeneralFit()
            
        return self.fitness, self.score

    def wallCollide(self, point) :
        if point[1] >= SCREEN_SIZE or point[1] < 0 or point[0] >= SCREEN_SIZE or point[0] < 0 :
            return True
        
        return False

    def bodyCollide(self, point) :
        if point.tolist() in self.player.posList[1:].tolist() :
            return True
        
        return False

    def itemCollide(self, point) :
        if all(point == self.item.position) :
            return True

        return False

    def get_inputs(self) :
        baseInput = np.zeros(shape = INPUT_SIZE)

        for i, direction in enumerate(DETECT_DIRS) :
            baseInput[i*3:i*3+3] = self.detection(direction)
        
        return baseInput

    def detection(self, direction) :
        sensingPoint = self.player.posList[0].copy()
        distance = 0

        detectItem = False
        detectBody = False

        sensingPoint += direction
        distance += 1

        detect = np.zeros(3)
        while not self.wallCollide(sensingPoint) :
            if not detectItem and self.itemCollide(sensingPoint) :
                detectItem = True
                detect[0] = 1

            if not detectBody and self.bodyCollide(sensingPoint) :
                detectBody = True
                detect[1] = 1

            sensingPoint += direction
            distance += 1
            
        detect[2] = 1 / distance
        
        return detect
    
    def calcGeneralFit(self) :
        fitness = (self.lifeTime) + ((2**self.score) + (self.score**2.1)*500) - (((.25 * self.lifeTime)**1.3) * (self.score**1.2))
        return round(fitness, 1)

class Snake :
    def __init__(self) :
        self.posList = np.array([[1, 2], [1, 1], [1, 0]])
        self.direction = RIGHT

    def nextHead(self) :
        prevHead = self.posList[0]
        moving = DIRECTIONS[self.direction]

        newHead = prevHead + moving

        return newHead

    def move(self) :
        self.posList = np.concatenate([[self.nextHead()], self.posList[:-1, :]], axis=0)

    def grow(self) :
        self.posList = np.concatenate([[self.nextHead()], self.posList], axis=0)

class Item :
    def __init__(self) :
        self.position = np.array([randint_20(), randint_20()])

    def newPosition(self) :
        self.position = np.array([randint_20(), randint_20()])

if __name__ == '__main__' :
    while True :
        game = SnakeGame()
        fitness, score = game.play()
        print('Fitness : {}, Score : {}'.format(fitness, score))
        