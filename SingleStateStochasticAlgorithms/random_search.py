
#! /usr/bin/env python
# -*- encoding: utf-8 -*-

__author__ = 'Ernesto Costa'
__date__ = 'February 2023'

import random
import matplotlib
import matplotlib.pyplot as plt

def random_search(domain,fitness,max_iter):
    """
    @domain = [...,[xmin_1, xmax_i],...]
    @fitness =  the quality of a candidate solution
    """
    best = random_indiv(domain)
    cost_best = fitness(best)
    data_best = [cost_best]
    for i in range(max_iter):
        candidate = random_indiv(domain)
        cost_candi = fitness(candidate)
        if cost_candi < cost_best: # minimization
            best = candidate
            cost_best = cost_candi
        data_best.append(cost_best)
    return data_best


def random_indiv(domain):
    individual = [random.uniform(domain_i[0], domain_i[1]) for domain_i in domain]
    return individual


# Example
def dejong_f1(individual):
    """ De Jong F1 or the sphere function"""
    return sum([x_i ** 2 for x_i in individual])

# Vizualize     
def vizualize_fitness(list_values):

    matplotlib.use('TkAgg')
    x = range(1,len(list_values)+1)
    y = list_values
    plt.title('Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.plot(x,y,'r-o')
    plt.show()
    #plt.savefig('random_search.png')
    
if __name__ == '__main__':
    dimensions = 3
    my_domain = [[-5.12,5.12] for i in range(dimensions)]
    data = random_search(my_domain,dejong_f1,150)
    print('Best: ', data[-1])
    vizualize_fitness(data)