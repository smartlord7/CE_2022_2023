"""
jb.py
The code below is given without any warranty!
Ernesto Costa, February, 2023
"""

import matplotlib.pyplot as plt
from erfa import num00b

from sea_bin_sol import *
from utils import *


# João Brandão

def fitness(indiv):
    return evaluate(phenotype(indiv), len(indiv))


def phenotype(indiv):
    fen = [i + 1 for i in range(len(indiv)) if indiv[i] == 1]
    return fen


def evaluate(indiv, comp):
    alfa = 1.0
    beta = 1.1
    return alfa * len(indiv) - beta * viola(indiv, comp)


def viola(indiv, comp):
    # Count violations
    v = 0
    for elem in indiv:
        limite = min(elem - 1, comp - elem)
        vi = 0
        for j in range(1, limite + 1):
            if ((elem - j) in indiv) and ((elem + j) in indiv):
                vi += 1
        v += vi
    return v


# PL3: TODO 3.4
def main():
    numb_generations = 100
    size_pop = 150
    size_cromo = 100
    prob_cross = 0.9
    tour_size = 3
    sel_parents = tour_sel(tour_size)
    mutation = muta_bin
    elite = 0.10
    sel_survivors = sel_survivors_elite(elite)
    fitness_func = fitness
    n_runs = 30
    probs_mut = [0.01, 0.05, 0.1, 0.2, 0.5]


    run_for_file('out.txt', n_runs, numb_generations, size_pop, size_cromo, prob_mut, prob_cross,,
                            sel_parents, one_point_cross, mutation, sel_survivors, fitness_func)



if __name__ == '__main__':
    """ To test the JBN function."""
    numb_generations = 100
    size_pop = 100
    size_cromo = 20
    prob_mut = 0.01
    prob_cross = 0.9
    tour_size = 3
    sel_parents = tour_sel(tour_size)
    recombination = uniform_cross
    mutation = muta_bin
    elite = 0.02
    sel_survivors = sel_survivors_elite(elite)
    fitness_func = fitness

    # example run
    best_1 = sea(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination, mutation,
                 sel_survivors, fitness_func)
    display(best_1, phenotype)

    seeds = list(range(30))
    boa, best_gener = run(3, numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination,
                          mutation, sel_survivors, fitness_func, seeds)
    display_stat_1(boa, best_gener)
