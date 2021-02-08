import random
from copy import deepcopy

import numpy as np
#from models import SnakeNet
from NewModel import SnakeNet

import pygame
from snake import *
import Snake_Statics as Static

pygame.init()

#nets = [SnakeNet(pre_weight='./Snake/BW.npy') for _ in range(N_POPULATION)]
nets = [SnakeNet() for _ in range(N_POPULATION)]
fitness_list = []
best = None
bw = None
bf = 0
bs = 0
generation = 0

try :
    while True :
        generation += 1
        
        if best is not None :
            nets.extend(best)

        for i, net in enumerate(nets) :
            game = SnakeGame(nn = net)
            fitness, score = game.play()
            bs = score if score > bs else bs

            net.fitness = fitness
            print('Gen {} Number {} of {} : Fitness {}, Score {}'.format(generation, i, len(nets), fitness, score))
        
        nets.sort(key = lambda x: x.fitness, reverse=True)
        print("Gen {}'s Best Fitness {} Score {}".format(generation, nets[0].fitness, bs))
        fitness_list.append(nets[0].fitness)

        best = deepcopy(nets[:N_BEST])
        if nets[0].fitness >= bf :
            bw = deepcopy(nets[0].weights)
            bf = nets[0].fitness

        for i in range(N_CHILDREN):
            aParent = random.choice(best)
            bParent = random.choice(best)
            child = aParent.crossover(bParent)

            best.append(child)

        nets = []
        for i in range(int(N_POPULATION / (N_BEST + N_CHILDREN))):
            for model in best:
                #child = model.mutate_best_base(best[0], MUTATION_RATE)
                #child = model.mutate_Gaussian(MUTATION_RATE)
                child = model.mutate_Random(MUTATION_RATE)
                nets.append(child)        

except Static.BreakException :
    pygame.quit()
    print(fitness_list)
    print('World Best Fitness : {}'.format(max(fitness_list)))

print('Save Best Weight ? (y/n)')
saved = input()

if saved == 'y' :
    np.save('./Snake/BW', bw)
