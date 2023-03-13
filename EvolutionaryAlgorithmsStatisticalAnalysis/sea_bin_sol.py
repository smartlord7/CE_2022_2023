"""
sea_bin.py
A very simple EA for binary representation.
Ernesto Costa, March 2015,February 2016,February 2019, March 2022
The code is provided with any warrenty. If you find a bug plese report it to
ernesto@dei.uc.pt
"""

__author__ = 'Ernesto Costa'
__date__ = 'March 2023'

import matplotlib.pyplot as plt
from utils import *
from random import random, randint, sample, seed
from operator import itemgetter


# 3.1 : TODO
# Plot best/ average
# Simple [Binary] Evolutionary Algorithm		
def sea_for_plot(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination, mutation,
                 sel_survivors, fitness_func):
    """Store the values of the best and of the average fitness for each generation."""
    # inicialize population: indiv = (cromo,fit)
    populacao = gera_pop(size_pop, size_cromo)
    # evaluate population
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
    best_gen = [best_pop(populacao)[1]]
    average_pop_gen = [average_pop(populacao)]
    for j in range(numb_generations):
        # sparents selection
        mate_pool = sel_parents(populacao)
        # Variation
        # ------ Crossover
        progenitores = []
        for i in range(0, size_pop - 1, 2):
            indiv_1 = mate_pool[i]
            indiv_2 = mate_pool[i + 1]
            filhos = recombination(indiv_1, indiv_2, prob_cross)
            progenitores.extend(filhos)
            # ------ Mutation
        descendentes = []
        for cromo, fit in progenitores:
            novo_indiv = mutation(cromo, prob_mut)
            descendentes.append((novo_indiv, fitness_func(novo_indiv)))
        # New population
        populacao = sel_survivors(populacao, descendentes)
        # Evaluate the new population
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
        best_gen.append(best_pop(populacao)[1])
        average_pop_gen.append(average_pop(populacao))
    return best_gen, average_pop_gen


# 3.2: TODO
# Run several times
def run(numb_runs, numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination, mutation,
        sel_survivors, fitness_func, seeds=None):
    """return the best over all for each generation togetther with the average of the averages of a population."""
    bests = []
    for i in range(numb_runs):
        if (not seeds is None):
            seed(seeds[i])
        best, aver_pop = sea_for_plot(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents,
                                      recombination, mutation, sel_survivors, fitness_func)
        bests.append(best)
    # transpose the matrix of results: from runs to generations
    stat_gener = list(zip(*bests))
    boa = [max(g_i) for g_i in stat_gener]  # maximization
    aver_bests_gener = [sum(g_i) / len(g_i) for g_i in stat_gener]
    return boa, aver_bests_gener


def run_best_at_the_end(numb_runs, numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents,
                        recombination, mutation, sel_survivors, fitness_func, seeds=None):
    """Return the best FITNESS at the end for each run."""
    bests = []
    for i in range(numb_runs):
        if (not seeds is None):
            seed(seeds[i])
        best_end = sea(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination,
                       mutation, sel_survivors, fitness_func)
        bests.append(best_end[1])
    return bests


# 3.3: TODO
# results in an external file
def run_for_file(filename, numb_runs, numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents,
                 recombination, mutation, sel_survivors, fitness_func, seeds=None):
    with open(filename, 'w') as f_out:
        for i in range(numb_runs):
            if (not seeds is None):
                seed(seeds[i])
            best = sea(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination,
                       mutation, sel_survivors, fitness_func)
            f_out.write(str(best[1]) + '\n')


def read_data_from_file(filename):
    with open(filename, 'r') as f_in:
        lines = f_in.readlines()
        values = [float(v) for v in lines]
        return values


def show_results(data):
    x = list(range(len(data)))
    plt.xlabel('Run')
    plt.ylabel('Best')
    plt.plot(x, data)
    plt.show()


# 3.4: TODO
# Best crossover operator

"""See in the jb.py file."""


# Simple [Binary] Evolutionary Algorithm		
def sea(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination, mutation,
        sel_survivors, fitness_func):
    # inicialize population: indiv = (cromo,fit)
    populacao = gera_pop(size_pop, size_cromo)
    # evaluate population
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
    for i in range(numb_generations):
        # sparents selection
        mate_pool = sel_parents(populacao)
        # Variation
        # ------ Crossover
        progenitores = []
        for i in range(0, size_pop - 1, 2):
            indiv_1 = mate_pool[i]
            indiv_2 = mate_pool[i + 1]
            filhos = recombination(indiv_1, indiv_2, prob_cross)
            progenitores.extend(filhos)
            # ------ Mutation
        descendentes = []
        for cromo, fit in progenitores:
            novo_indiv = mutation(cromo, prob_mut)
            descendentes.append((novo_indiv, fitness_func(novo_indiv)))
        # New population
        populacao = sel_survivors(populacao, descendentes)
        # Evaluate the new population
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
    return best_pop(populacao)


# Initialize population
def gera_pop(size_pop, size_cromo):
    return [(gera_indiv(size_cromo), 0) for i in range(size_pop)]


def gera_indiv(size_cromo):
    # random initialization
    indiv = [randint(0, 1) for i in range(size_cromo)]
    return indiv


# Variation operators: Binary mutation
def muta_bin(indiv, prob_muta):
    # Mutation by gene
    cromo = indiv[:]
    for i in range(len(indiv)):
        cromo[i] = muta_bin_gene(cromo[i], prob_muta)
    return cromo


def muta_bin_gene(gene, prob_muta):
    g = gene
    value = random()
    if value < prob_muta:
        g ^= 1
    return g


# Variation Operators :Crossover
def one_point_cross(indiv_1, indiv_2, prob_cross):
    value = random()
    if value < prob_cross:
        cromo_1 = indiv_1[0]
        cromo_2 = indiv_2[0]
        pos = randint(0, len(cromo_1))
        f1 = cromo_1[0:pos] + cromo_2[pos:]
        f2 = cromo_2[0:pos] + cromo_1[pos:]
        return ((f1, 0), (f2, 0))
    else:
        return (indiv_1, indiv_2)


def two_points_cross(indiv_1, indiv_2, prob_cross):
    value = random()
    if value < prob_cross:
        cromo_1 = indiv_1[0]
        cromo_2 = indiv_2[0]
        pc = sample(range(len(cromo_1)), 2)
        pc.sort()
        pc1, pc2 = pc
        f1 = cromo_1[:pc1] + cromo_2[pc1:pc2] + cromo_1[pc2:]
        f2 = cromo_2[:pc1] + cromo_1[pc1:pc2] + cromo_2[pc2:]
        return ((f1, 0), (f2, 0))
    else:
        return (indiv_1, indiv_2)


def uniform_cross(indiv_1, indiv_2, prob_cross):
    value = random()
    if value < prob_cross:
        cromo_1 = indiv_1[0]
        cromo_2 = indiv_2[0]
        f1 = []
        f2 = []
        for i in range(0, len(cromo_1)):
            if random() < 0.5:
                f1.append(cromo_1[i])
                f2.append(cromo_2[i])
            else:
                f1.append(cromo_2[i])
                f2.append(cromo_1[i])
        return ((f1, 0), (f2, 0))
    else:
        return (indiv_1, indiv_2)


# Parents Selection: tournament
def tour_sel(t_size):
    def tournament(pop):
        size_pop = len(pop)
        mate_pool = []
        for i in range(size_pop):
            winner = one_tour(pop, t_size)
            mate_pool.append(winner)
        return mate_pool

    return tournament


def one_tour(population, size):
    """Maximization Problem. Deterministic"""
    pool = sample(population, size)
    pool.sort(key=itemgetter(1), reverse=True)
    return pool[0]


# Survivals Selection: elitism
def sel_survivors_elite(elite):
    def elitism(parents, offspring):
        size = len(parents)
        comp_elite = int(size * elite)
        offspring.sort(key=itemgetter(1), reverse=True)
        parents.sort(key=itemgetter(1), reverse=True)
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population

    return elitism


# Auxiliary
def display(indiv, phenotype):
    print('Chromo: %s\nFitness: %s' % (phenotype(indiv[0]), indiv[1]))


def best_pop(populacao):
    populacao.sort(key=itemgetter(1), reverse=True)
    return populacao[0]


# New
def average_pop(populacao):
    return sum([indiv[1] for indiv in populacao]) / len(populacao)


# -------------------  Problem Specific Definitions  ------------  
# -------------------  One max problem --------------------------

def merito(indiv):
    # wrapper for fitness evaluation
    return evaluate(fenotipo(indiv))


def fenotipo(indiv):
    return indiv


def evaluate(indiv):
    return sum(indiv)


if __name__ == '__main__':
    """to test the code with oneMax function."""
    numb_generations = 100
    size_pop = 50
    size_cromo = 20
    prob_mut = 0.01
    prob_cross = 0.9
    sel_parents = tour_sel(3)
    recombination = uniform_cross
    mutation = muta_bin
    sel_survivors = sel_survivors_elite(0.02)
    fitness_func = merito
    # best_1 = sea(numb_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func)
    # display(best_1,fenotipo)

    # best_gen, average_pop_gen = sea_for_plot(numb_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func)
    # display_stat_1(best_gen,average_pop_gen)

    num_runs = 10
    # boa, average_best_gen = run(num_runs,numb_generations,size_pop, size_cromo, prob_mut,prob_cross,sel_parents,recombination,mutation,sel_survivors, fitness_func)
    # display_stat_1(boa,average_best_gen)

    filename = 'test_sea.txt'
    run_for_file(filename, num_runs, numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents,
                 recombination, mutation, sel_survivors, fitness_func)
    results = read_data_from_file(filename)
    show_results(results)
