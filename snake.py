import os, random
from datetime import timedelta, datetime
from time import sleep
import math

import numpy as np
import pygame

from Snake_Statics import *

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
                inputs = self.get_inputs()
                outputs = self.nn.forward(inputs)
                result = np.argsort(outputs)
                
                #print(outputs)
                for possible in result :
                    if self.player.direction + MOVE_NN_DIR[possible] != MOVE_LIMIT :
                        self.player.direction = MOVE_NN_DIR[possible]
                        lastCtrlTime = datetime.now()
                        break
                        
            if all(self.player.posList[0] == self.item.position) :
                self.player.grow()
                self.score += 1
                self.lifeLeft = LIFE_LEFT
                self.item.newPosition()

                while self.item.position in self.player.posList :
                    self.item.newPosition()

                if self.distanceFit :
                    self.fitness += 30
            else :
                self.player.move()
                self.lifeLeft -= 1

            self.lifeTime += 1

            if self.player.posList[0].tolist() in self.player.posList[1:].tolist() :
                self.done = True

            if self.wallCollide(self.player.posList[0]) :
                self.done = True

            if self.lifeLeft <= 0 and self.nn is not None:
                self.done = True

            if self.distanceFit == 0 :
                self.fitness = self.calcTimeFit()
            elif self.distanceFit == 1 :
                self.fitness = self.calcDistancefit(self.fitness)
            else :
                self.fitness = self.calcGeneralFit()
            
            self.item.draw(self.screen)
            self.player.draw(self.screen)
            
            scoreFont = self.font.render(str(self.score), True, WHITE)
            self.screen.blit(scoreFont, (5,5))

            pygame.display.update()

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
        
        head = self.player.posList[0]
        
        # Straight
        #if np.any(head == self.item.position) and np.sum(head * DIRECTIONS[self.player.direction]) <= \
        #   np.sum(self.item.position * DIRECTIONS[self.player.direction]) :
        #    baseInput[24] = 1

        #print(baseInput)
        return baseInput

    def detection(self, direction) :
        sensingPoint = self.player.posList[0]
        distance = 0

        detectItem = False
        detectBody = False

        sensingPoint = sensingPoint + direction
        distance += 1

        detect = np.zeros(3)
        while(not self.wallCollide(sensingPoint)) :
            color = None
            rect = pygame.Rect((sensingPoint[1] * PIXEL_SIZE, sensingPoint[0] * PIXEL_SIZE), (5,5))

            if(not detectItem and self.itemCollide(sensingPoint)) :
                detectItem = True
                detect[0] = 1
                color = YELLOW
            
            if(not detectBody and self.bodyCollide(sensingPoint)) :
                detectBody = True
                detect[1] = 1
                color = RED

            if SEE_VISION :
                if color is None :
                    pygame.draw.rect(self.screen, WHITE, rect)
                else :
                    pygame.draw.rect(self.screen, color, rect)

            sensingPoint = sensingPoint + direction
            distance += 1
            
        detect[2] = 1 / distance

        if SEE_VISION :
            head = pygame.Rect((self.player.posList[0][1] * PIXEL_SIZE, self.player.posList[0][0] * PIXEL_SIZE), (5,5))
            pygame.draw.rect(self.screen, YELLOW, head)

        return detect

    def calcGeneralFit(self) :
        fitness = (self.lifeTime) + ((2**self.score) + (self.score**2.1)*500) - (((.25 * self.lifeTime)**1.3) * (self.score**1.2))
        return round(fitness, 1)

    def calcTimeFit(self) :
        if self.score < 10 :
            fitness = math.floor(self.lifeTime * self.lifeTime) * math.pow(2, self.score)
        else :
            fitness = math.floor(self.lifeTime * self.lifeTime) * math.pow(2, 10) * (self.score - 9)

        return fitness

    def calcDistancefit(self, fitness) :
        current_distance = np.linalg.norm(self.player.posList[0] - self.item.position)

        if self.last_distance > current_distance :
            fitness += 1.
        else :
            fitness -= 1.5

        self.last_distance = current_distance

        return fitness

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

