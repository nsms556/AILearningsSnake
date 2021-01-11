import random
from copy import deepcopy

import numpy as np
from models import SnakeNet

import pygame
from snake import *
import Snake_Statics as Static

pygame.init()

nets = [SnakeNet(pre_weight='BW0.npy') for _ in range(N_POPULATION)]
best = None
fitness_list = []

generation = 0

try :
    while True :
        generation += 1
        
        for i, net in enumerate(nets) :
            game = SnakeGame(nn = net)
            fitness, score = game.play()

            #if score == 0 and fitness > 10000 :
            #    fitness /= 10
            net.fitness = fitness
            print('Gen {} : Net Number {} of {} -> Fitness : {}, Score {}'.format(generation, i, len(nets), fitness, score))
        
        if best is not None :
            nets.extend(best)
        
        nets.sort(key = lambda x: x.fitness, reverse=True)
        print("Gen {}'s Best Fitness {}".format(generation, nets[0].fitness))
        fitness_list.append(nets[0].fitness)
        
        #print("Present Best Net's Weight")
        #print(nets[0].w1, nets[0].w2)

        best = deepcopy(nets[:N_BEST])

        for i in range(N_CHILDREN):
            aParent = best[0]
            bParent = random.choice(best[1:N_BEST])
            child = aParent.crossover(bParent)

            best.append(child)

        nets = []
        for i in range(int(N_POPULATION / (N_BEST + N_CHILDREN))):
            for model in best:
                child = model.mutate(best[0], MUTATION_RATE)
                nets.append(child)        

except Static.BreakException :
    pygame.quit()
    print(fitness_list)
    print('World Best Fitness : {}'.format(max(fitness_list)))
'''
print('Save Best Weight ? (y/n)')
saved = input()

if saved == 'y' :
    np.save('./Snake/BW', best[0].weights)
'''