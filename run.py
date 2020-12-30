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
            new_genome = deepcopy(best[0])
            a_genome = random.choice(best)
            b_genome = random.choice(best)

            cut = random.randint(0, new_genome.w1.shape[1])
            new_genome.w1[i, :cut] = a_genome.w1[i, :cut]
            new_genome.w1[i, cut:] = b_genome.w1[i, cut:]

            cut = random.randint(0, new_genome.w2.shape[1])
            new_genome.w2[i, :cut] = a_genome.w2[i, :cut]
            new_genome.w2[i, cut:] = b_genome.w2[i, cut:]

            cut = random.randint(0, new_genome.w3.shape[1])
            new_genome.w3[i, :cut] = a_genome.w3[i, :cut]
            new_genome.w3[i, cut:] = b_genome.w3[i, cut:]

            best.append(new_genome)

        nets = []
        for i in range(int(N_POPULATION / (N_BEST + N_CHILDREN))):
            for bg in best:
                new_genome = deepcopy(bg)

                mean = 20
                stddev = 10

                if random.uniform(0, 1) < MUTATION_RATE:
                    new_genome.w1 += new_genome.w1 * np.random.normal(mean, stddev, size=(24, 10)) / 100 * np.random.randint(-1, 2, (24, 10))
                if random.uniform(0, 1) < MUTATION_RATE:
                    new_genome.w2 += new_genome.w2 * np.random.normal(mean, stddev, size=(10, 10)) / 100 * np.random.randint(-1, 2, (10, 10))
                if random.uniform(0, 1) < MUTATION_RATE:
                    new_genome.w3 += new_genome.w3 * np.random.normal(mean, stddev, size=(10, 4)) / 100 * np.random.randint(-1, 2, (10, 4))
                
                nets.append(new_genome)        

except Static.BreakException :
    pygame.quit()
    print(fitness_list)
    print('World Best Fitness : {}'.format(max(fitness_list)))
