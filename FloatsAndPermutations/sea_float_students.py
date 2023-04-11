#! /usr/bin/env python

"""
sea_float.py
A very simple EA for float representation.
Ernesto Costa, February 2022
"""

__author__ = 'Ernesto Costa'
__date__ = 'February 2023'

import numpy as np
import matplotlib.pyplot as plt
from random import random, randint, uniform, sample, shuffle, gauss, seed
from operator import itemgetter


# run
def run_best_at_the_end_float(numb_runs, numb_generations, size_pop, domain, prob_mut, sigma, prob_cross, sel_parents,
                              recombination, mutation, sel_survivors, fitness_func, seeds):
    """Return the best FITNESS at the end for each run, as well as the gen at which that individual was found."""
    bests = []
    gens = []
    for i in range(numb_runs):
        print('\t----run ' + str(i) + '----')
        seed(seeds[i])
        best_end, g = sea_float(numb_generations, size_pop, domain, prob_mut, sigma, prob_cross, sel_parents,
                                recombination, mutation, sel_survivors, fitness_func)
        bests.append(best_end[1])
        gens.append(g)
        print('\t\tbest ' + str(best_end[1]) + ', gen ' + str(g))
    return bests, gens


# Simple [Float] Evolutionary Algorithm		
def sea_float(numb_generations, size_pop, domain, prob_mut, sigma, prob_cross, sel_parents, recombination, mutation,
              sel_survivors, fitness_func):
    """
    inicialize population: indiv = (cromo,fit)
    domain = [...-,[inf_i, sup_i],...]
    sigma = [..., sigma_i, ...]
    """

    populacao = gera_pop(size_pop, domain)
    # evaluate population
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
    generation = 0
    best = best_pop(populacao)
    for gen in range(numb_generations):
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
            if (len(progenitores) == size_pop):
                break
        # ------ Mutation
        descendentes = []
        for cromo, fit in progenitores:
            novo_indiv = mutation(cromo, prob_mut, domain)
            descendentes.append((novo_indiv, fitness_func(novo_indiv)))
        # New population
        populacao = sel_survivors(populacao, descendentes)
        # Evaluate the new population
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]
        new_best = best_pop(populacao)
        if new_best[1] < best[1]:
            generation = gen
            best = new_best
    return best_pop(populacao), generation


# Initialize population
def gera_pop(size_pop, domain):
    return [(gera_indiv_float(domain), 0) for i in range(size_pop)]


def gera_indiv_float(domain):
    return [uniform(domain[i][0], domain[i][1]) for i in range(len(domain))]


# Variation operators: ------ > gaussian float mutation	    
def muta_gaussian(sigma):
    def muta_float_gaussian(cromo, prob_muta, domain):
        for i in range(len(cromo)):
            cromo[i] = muta_float_gene(cromo[i], prob_muta, domain[i], sigma[i])
        return cromo

    return muta_float_gaussian


def muta_float_gene(gene, prob_muta, domain_i, sigma_i):
    value = random()
    new_gene = gene
    if value < prob_muta:
        muta_value = gauss(0, sigma_i)
        new_gene = gene + muta_value
        if new_gene < domain_i[0]:
            new_gene = domain_i[0]
        elif new_gene > domain_i[1]:
            new_gene = domain_i[1]
    return new_gene


# TODO: implement uniform mutation
def uniform_mutation(individual, mutation_prob):
    mutated_individual = []
    for gene in individual:
        if random() < mutation_prob:
            # Generate a random value for the gene
            new_gene = uniform(0, 1)
        else:
            new_gene = gene
        mutated_individual.append(new_gene)
    return mutated_individual


# Variation Operators : Aritmetical  Crossover
def cross(alpha):
    def aritmetical_cross(indiv_1, indiv_2, prob_cross):
        size = len(indiv_1[0])
        value = random()
        if value < prob_cross:
            cromo_1 = indiv_1[0]
            cromo_2 = indiv_2[0]
            f1 = [None] * size
            f2 = [None] * size
            for i in range(size):
                f1[i] = alpha * cromo_1[i] + (1 - alpha) * cromo_2[i]
                f2[i] = (1 - alpha) * cromo_1[i] + alpha * cromo_2[i]
            return ((f1, 0), (f2, 0))
        return indiv_1, indiv_2

    return aritmetical_cross


# TODO: implement the heuristic crossover

def heuristic_crossover(parent1, parent2):
    # Create a new child with the same length as the parents
    child = [None] * len(parent1)

    # Iterate over each gene in the child
    for i in range(len(child)):
        if parent1[i] == parent2[i]:
            # If the genes are equal, the child's gene is the same
            child[i] = parent1[i]
        else:
            # Otherwise, the child's gene is a weighted average of the parents' genes
            alpha = uniform(0, 1)
            child[i] = (1 - alpha) * parent1[i] + alpha * parent2[i]

    return child

# Tournament Selection
def tour_sel(t_size):
    def tournament(pop):
        size_pop = len(pop)
        mate_pool = []
        for i in range(size_pop):
            winner = tour(pop, t_size)
            mate_pool.append(winner)
        return mate_pool

    return tournament


def tour(population, size):
    """Minimization Problem.Deterministic"""
    pool = sample(population, size)
    pool.sort(key=itemgetter(1))
    return pool[0]


# Survivals: elitism
def sel_survivors_elite(elite):
    def elitism(parents, offspring):
        """Minimization."""
        size = len(parents)
        comp_elite = int(size * elite)
        offspring.sort(key=itemgetter(1))
        parents.sort(key=itemgetter(1))
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population

    return elitism


# Auxiliary
def best_pop(populacao):
    """Minimization."""
    populacao.sort(key=itemgetter(1))
    return populacao[0]


def average_pop(populacao):
    return sum([fit for cromo, fit in populacao]) / len(populacao)


if __name__ == '__main__':
    """The code for an EA working with floats."""

    pass
