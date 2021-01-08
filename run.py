import random
from copy import deepcopy

import numpy as np
from models import SnakeNet

import pygame
from snake import *
import Snake_Statics as Static

pygame.init()

nets = [SnakeNet() for _ in range(N_POPULATION)]
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
                '''
                new_genome = deepcopy(bg)

                mean = 20
                stddev = 10

                if random.uniform(0, 1) < MUTATION_RATE:
                    new_genome.w1 += new_genome.w1 * np.random.normal(mean, stddev, size=(Static.INPUT_SIZE, Static.HIDDEN_SIZE)) / 100 * np.random.randint(-1, 2, (Static.INPUT_SIZE, Static.HIDDEN_SIZE))
                if random.uniform(0, 1) < MUTATION_RATE:
                    new_genome.w2 += new_genome.w2 * np.random.normal(mean, stddev, size=(Static.HIDDEN_SIZE, Static.HIDDEN_SIZE)) / 100 * np.random.randint(-1, 2, (Static.HIDDEN_SIZE, Static.HIDDEN_SIZE))
                if random.uniform(0, 1) < MUTATION_RATE:
                    new_genome.w3 += new_genome.w3 * np.random.normal(mean, stddev, size=(Static.HIDDEN_SIZE, Static.OUTPUT_SIZE)) / 100 * np.random.randint(-1, 2, (Static.HIDDEN_SIZE, Static.OUTPUT_SIZE))
                '''

                child = model.mutate(best[0], MUTATION_RATE)
                nets.append(child)        

except Static.BreakException :
    pygame.quit()
    print(fitness_list)
    print('World Best Fitness : {}'.format(max(fitness_list)))
'''
print('Save Best Weight ? (y/n)')y
saved = input()

if saved == 'y' :
    np.savez('BW', best[0])
'''