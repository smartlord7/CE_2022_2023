"""
sea_bin.py
A very simple EA for binary representation.
Ernesto Costa, March 2015,February 2016, February 2023
"""

__author__ = 'Ernesto Costa'
__date__ = 'February 2023'

import matplotlib.pyplot as plt

from random import random,randint, sample
from operator import itemgetter
from utils import *
import numpy as np

def show_results(data):
    x = list(range(len(data)))
    y = [indiv[1] for indiv in data]
    plt.xlabel('Run')
    plt.ylabel('Best')
    plt.plot(x,y)
    plt.show()
    

# Simple [Binary] Evolutionary Algorithm		
def sea(numb_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func):
    # inicialize population: indiv = (cromo,fit)
    population = gera_pop(size_pop,size_cromo)
    # evaluate population
    population = [(indiv[0], fitness_func(indiv[0])) for indiv in population]
    
    avg_fitness = []
    best_fitness = []
    for i in range(numb_generations):
        curr_avg_fitness = sum([individual_data[1] for individual_data in population]) / size_pop
        avg_fitness.append(curr_avg_fitness)
        curr_best_fitness = best_pop(population)
        best_fitness.append(curr_best_fitness[1])
        # parents selection
        mate_pool = sel_parents(population)
    # Variation
    # ------ Crossover
        progenitores = []
        for i in  range(0,size_pop-1,2):
            indiv_1= mate_pool[i]
            indiv_2 = mate_pool[i+1]
            filhos = recombination(indiv_1,indiv_2, prob_cross)
            progenitores.extend(filhos) 
        # ------ Mutation
        descendentes = []
        for cromo,fit in progenitores:
            novo_indiv = mutation(cromo,prob_mut)
            descendentes.append((novo_indiv,fitness_func(novo_indiv)))
        # New population
        population = sel_survivors(population,descendentes)
        # Evaluate the new population
        population = [(indiv[0], fitness_func(indiv[0])) for indiv in population]     
    return avg_fitness, best_fitness, best_pop(population)


# Initialize population
def gera_pop(size_pop,size_cromo):
    return [(gera_indiv(size_cromo),0) for i in range(size_pop)]


def gera_indiv(size_cromo):
    # random initialization
    indiv = [randint(0,1) for i in range(size_cromo)]
    return indiv


# Variation operators: Binary mutation	    
def muta_bin(indiv,prob_muta):
    # Mutation by gene
    cromo = indiv[:]
    for i in range(len(indiv)):
        cromo[i] = muta_bin_gene(cromo[i],prob_muta)
    return cromo


def muta_bin_gene(gene, prob_muta):
    g = gene
    value = random()
    if value < prob_muta:
        g ^= 1
    return g


# Variation Operators :Crossover
def one_point_cross(indiv_1, indiv_2,prob_cross):
    value = random()
    if value < prob_cross:
        cromo_1 = indiv_1[0]
        cromo_2 = indiv_2[0]
        pos = randint(0,len(cromo_1))
        f1 = cromo_1[0:pos] + cromo_2[pos:]
        f2 = cromo_2[0:pos] + cromo_1[pos:]
        return ((f1,0),(f2,0))
    else:
        return (indiv_1,indiv_2)


def two_points_cross(indiv_1, indiv_2,prob_cross):
    value = random()
    if value < prob_cross:
        cromo_1 = indiv_1[0]
        cromo_2 = indiv_2[0]
        pc= sample(range(len(cromo_1)),2)
        pc.sort()
        pc1,pc2 = pc
        f1= cromo_1[:pc1] + cromo_2[pc1:pc2] + cromo_1[pc2:]
        f2= cromo_2[:pc1] + cromo_1[pc1:pc2] + cromo_2[pc2:]
        return ((f1,0),(f2,0))
    else:
        return (indiv_1,indiv_2)


def uniform_cross(indiv_1, indiv_2,prob_cross):
    value = random()
    if value < prob_cross:
        cromo_1 = indiv_1[0]
        cromo_2 = indiv_2[0]
        f1=[]
        f2=[]
        for i in range(0,len(cromo_1)):
            if random() < 0.5:
                f1.append(cromo_1[i])
                f2.append(cromo_2[i])
            else:
                f1.append(cromo_2[i])
                f2.append(cromo_1[i])
        return ((f1,0),(f2,0))
    else:
        return (indiv_1,indiv_2)


# Parents Selection: tournament
def tour_sel(t_size):
    def tournament(pop):
        size_pop= len(pop)
        mate_pool = []
        for i in range(size_pop):
            winner = one_tour(pop,t_size)
            mate_pool.append(winner)
        return mate_pool
    return tournament


def one_tour(population,size):
    """Maximization Problem. Deterministic"""
    pool = sample(population, size)
    pool.sort(key=itemgetter(1), reverse=True)
    return pool[0]


# Survivals Selection: elitism
def sel_survivors_elite(elite):
    def elitism(parents,offspring):
        size = len(parents)
        comp_elite = int(size* elite)
        offspring.sort(key=itemgetter(1), reverse=True)
        parents.sort(key=itemgetter(1), reverse=True)
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population
    return elitism


# Auxiliary
    
def best_pop(population):
    population.sort(key=itemgetter(1),reverse=True)
    return population[0]

    
# -------------------  Problem Specific Definitions  ------------  
# -------------------  One max problem --------------------------

def merito(indiv):
    # wrapper for fitness evaluation
    return evaluate(fenotipo(indiv))


def fenotipo(indiv):
    return indiv


def evaluate(indiv):
    return sum(indiv)

# --------------------- TODO -----------------------------------

# 3.1: Change parameters and plot (best, average population)

# 3.2: Run several times and plot (best, average population)

# 3.3: Store in an external file (best at the end, one per line)

# 3.4: Which crossover is best (one-point, two points, uniform)


if __name__ == '__main__':
    #to test the code with oneMax function
    best_1 = sea(100, 20,100,0.01,0.9,tour_sel(3),uniform_cross,muta_bin,sel_survivors_elite(0.02), merito)
    display(best_1,fenotipo)
