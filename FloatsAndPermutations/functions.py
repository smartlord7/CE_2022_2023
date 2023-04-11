"""
functions.py
Examples for function optimizattion.
"""

__author__ = 'Ernesto Costa'
__date__ = 'March 2023'

import math
from random import uniform
from sea_float_students import *
from utils import *


# Fitness
def merito_r(indiv):
    return rastrigin(fenotipo(indiv))


def merito_s(indiv):
    return schwefel(fenotipo(indiv))


def merito_q(indiv):
    return quartic(fenotipo(indiv))


def fenotipo(indiv):
    return indiv


def rastrigin(indiv):
    """
    rastrigin function
    domain = [-5.12, 5.12]
    minimum at (0,....,0)
    """
    n = len(indiv)
    A = 10
    return A * n + sum([x ** 2 - A * math.cos(2 * math.pi * x) for x in indiv])


def schwefel(indiv):
    """
    schwefel function
    domain = [-500; 500]
    minimum at (420.9687,...,420.9687)
    """
    y = sum([-x * math.sin(math.sqrt(math.fabs(x))) for x in indiv])
    return y


def quartic(indiv):
    """
    quartic = DeJong 4
    domain = [-1.28; 1.28]
    minimum 0 at x = 0
    """
    y = sum([(i + 1) * x for i, x in enumerate(indiv)]) + uniform(0, 1)
    return y


if __name__ == '__main__':
    # HERE: choose the benchmark problem, define the parameters and run for file!
    dimensions = 5
    sigma_r = [0.1 for i in range(dimensions)]
    sigma = sigma_r
    fitness_func = merito_r
    alpha = 0.4
    cross_operator = cross(alpha)
    mutation = muta_gaussian(sigma)
    numb_runs = 5
    numb_generations = 150
    size_pop = 150

    # domain is dependent on the chosen function...
    # Rastrigin       
    domain_r = [[-5.12, 5.12]] * dimensions
    domain = domain_r

    prob_mut = 0.01
    prob_cross = 0.7
    tour_size = 3
    sel_parents = tour_sel(tour_size)
    elite = 0.1
    sel_survivors = sel_survivors_elite(elite)

    seeds = [830, 859, 782, 258, 307, 950, 497, 753, 303, 758, 849, 505, 170, 568, 734, 672, 740, 836, 122, 275, 930,
             244, 463, 162, 744, 317, 936, 236, 273, 156]
    numb_runs = len(seeds)

    best, g = run_best_at_the_end_float(numb_runs, numb_generations, size_pop, domain, prob_mut, sigma, prob_cross,
                                        sel_parents, cross_operator, mutation, sel_survivors, fitness_func, seeds)

    import matplotlib.pyplot as plt

    plt.subplot(211)
    plt.plot(best)
    plt.subplot(212)
    plt.plot(g)
    plt.show()
