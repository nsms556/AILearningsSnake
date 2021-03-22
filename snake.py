import os, random
from datetime import timedelta, datetime
from time import sleep
import math

import numpy as np
import pygame

from utils.Snake_Statics import *
import utils.utils as U
import utils.AIUtils as AU

class SnakeGame :
    def __init__(self, nn=None) :
        self.screen = pygame.display.set_mode([DISPLAY_WIDTH, DISPLAY_HEIGHT])
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('malgungothic', 20)

        self.done = False
        
        self.nn = nn
        self.distanceFit = SELECT_FIT
        
    def play(self) :
        self.player = Snake()
        self.item = Item()
        lastCtrlTime = datetime.now()

        FPS = HM_FPS if self.nn is None else NN_FPS

        self.score = 0
        self.fitness = 0
        self.last_distance = np.inf
        self.lastItemTime = 0
        self.lifeTime = 0
        self.lifeLeft = LIFE_LEFT

        while not self.done :
            self.clock.tick(FPS)
            self.screen.fill(BLACK)

            for event in pygame.event.get() :
                if event.type == pygame.QUIT :
                    raise BreakException

                if event.type == pygame.KEYDOWN :
                    if event.key == pygame.K_ESCAPE :
                        raise BreakException
                    
                    if event.key in MOVE_DIR :
                        if MOVE_DIR[event.key] + self.player.direction != MOVE_LIMIT and timedelta(seconds=0.1) <= datetime.now() - lastCtrlTime :
                            self.player.direction = MOVE_DIR[event.key]
                            lastCtrlTime = datetime.now()
                    elif event.key == pygame.K_SPACE :
                        pause = True

                        while pause :
                            for ee in pygame.event.get() :
                                if ee.type == pygame.QUIT :
                                    raise BreakException
                                elif ee.type == pygame.KEYDOWN :
                                    if ee.key == pygame.K_SPACE :
                                        pause = False
                                    elif ee.key == pygame.K_ESCAPE :
                                        raise BreakException
                                    elif ee.key == pygame.K_p :
                                        print('Inputs')
                                        print(inputs)
                                        print('Outputs')
                                        print(outputs)
                                        print('Head')
                                        print(self.player.posList[0])
                                        print('Item')
                                        print(self.item.position)
            
            if self.nn != None :
                inputs = AU.get_inputs(self.player.posList, self.item.position, INPUT_SIZE, DETECT_DIRS)
                outputs = self.nn.forward(inputs)
                result = np.argsort(outputs)
                
                #print(outputs)
                for possible in result :
                    if self.player.direction + MOVE_NN_DIR[possible] != MOVE_LIMIT :
                        self.player.direction = MOVE_NN_DIR[possible]
                        lastCtrlTime = datetime.now()
                        break
                        
            if U.itemCollide(self.player.posList[0], self.item.position) :
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

            if U.bodyCollide(self.player.posList[0], self.player.posList) :
                self.done = True

            if U.wallCollide(self.player.posList[0]) :
                self.done = True

            if self.lifeLeft <= 0 and self.nn is not None:
                self.done = True

            self.fitness = U.calcFitness(self.lifeTime, self.score)
            
            self.item.draw(self.screen)
            self.player.draw(self.screen)
            
            scoreFont = self.font.render(str(self.score), True, WHITE)
            self.screen.blit(scoreFont, (5,5))

            pygame.display.update()

        return self.fitness, self.score

class Snake :
    def __init__(self) :
        self.posList = np.array([[1, 2], [1, 1], [1, 0]])
        self.direction = RIGHT

    def draw(self, screen) :
        for position in self.posList :
            drawBlock(screen, WHITE, position)
    
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

    def draw(self, screen) :
        drawBlock(screen, GREEN, self.position)

    def newPosition(self) :
        self.position = np.array([randint_20(), randint_20()])

if __name__ == '__main__' :
    pygame.init()

    try :
        while True :
            game = SnakeGame()
            fitness, score = game.play()
            print('Fitness : {}, Score : {}'.format(fitness, score))
    except BreakException :
        pygame.quit()

