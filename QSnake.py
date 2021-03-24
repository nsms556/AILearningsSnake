import os, random
from datetime import timedelta, datetime
from time import sleep
import math

import numpy as np
import pygame
from pygame.constants import KEYDOWN

from utils.Snake_Statics import *
import utils.utils as U

class SnakeGame :
    def __init__(self) :
        self.screen = pygame.display.set_mode([DISPLAY_WIDTH, DISPLAY_HEIGHT])
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('malgungothic', 20)

        self.done = False
    
    def initialize(self) :
        self.player = Snake()
        self.item = Item()
        
        self.score = 0
        self.placeItem()
        self.FPS = NN_FPS
        self.lifeLeft = NN_FPS * 5
        self.lifeTime = 0

    def play_step(self) :
        self.lifeTime += 1

        for event in pygame.event.get() :
            if event.type == pygame.QUIT :
                raise BreakException

        reward = 0
        done = False

        head = self.player.posList[0]

        if U.wallCollide(head) or U.bodyCollide(head, self.player.posList) or self.lifeLeft < 1 :
            done = True
            reward = -10

            return reward, done, self.score

        if U.itemCollide(head, self.item.position) :
            self.score += 1
            reward = 10

            self.player.grow()
            self.placeItem()

            self.lifeLeft = NN_FPS * (int(self.score ** 0.5) + 5)
        else :
            self.player.move()

            self.lifeLeft -= 1

        self.updateUI()
        self.clock.tick(self.FPS)

        return reward, done, self.score

    def play(self) :
        self.player = Snake()
        self.item = Item()

        self.score = 0
        self.reward = 0
        self.last_distance = np.inf
        self.lastItemTime = 0
        self.lifeTime = 0
        self.lifeLeft = LIFE_LEFT

        while not self.done :
            self.clock.tick(FPS)
            self.screen.fill(BLACK)
           
            if U.itemCollide(self.player.posList[0], self.item.position) :
                self.player.grow()
                self.score += 1
                self.lifeLeft = LIFE_LEFT + (self.score / 2 * NN_FPS)
                
                self.placeItem()

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
    
            self.item.draw(self.screen)
            self.player.draw(self.screen)
            
            scoreFont = self.font.render(str(self.score), True, WHITE)
            self.screen.blit(scoreFont, (5,5))

            pygame.display.update()

        return self.fitness, self.score

    def placeItem(self) :
        self.item.newPosition()
        while self.item.position.tolist() in self.player.posList.tolist() :
            self.item.newPosition()

    def updateUI(self) :
        self.screen.fill(BLACK)

        self.item.draw(self.screen)
        self.player.draw(self.screen)

        scoreFont = self.font.render(str(self.score), True, WHITE)
        self.screen.blit(scoreFont, (5,5))

        pygame.display.update()


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

            game.initialize()
            
            while True :
                reward, done, score = game.play_step()

                if done == True :
                    break
            
            print('Score : {}, reward : {}'.format(score, reward))

    except BreakException :
        pygame.quit()

