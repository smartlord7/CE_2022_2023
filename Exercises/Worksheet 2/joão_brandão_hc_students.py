"""
Números de João Brandão.

Algoritmo: Hill-climbing
Pertubation: best neighbor
Representation: binary

"""

__author__ = 'Ernesto Costa'
__date__ = 'February 2023'


import random
import matplotlib
import matplotlib.pyplot as plt

def display_data(data):
    """Plot the data"""
    x = list(range(len(data)))
    plt.grid(True)
    plt.plot(x,data, 'r')
    plt.show()

# Basic Hill Climbing
def jb_hc(problem_size, max_iter,fitness):
    candidate = random_indiv(problem_size)
    cost_candi = fitness(candidate)
    list_of_best = [cost_candi]
    for i in range(max_iter):
        next_neighbor = best_neighbor(candidate,fitness)
        cost_next_neighbor = fitness(next_neighbor)
        if cost_next_neighbor >= cost_candi: 
            candidate = next_neighbor
            cost_candi = cost_next_neighbor 
        list_of_best.append(cost_candi)
    display_data(list_of_best)
    return candidate
     
# Random Individual
def random_indiv(size):
    return [random.randint(0,1) for i in range(size)]

# Best neighbor
def best_neighbor(individual, fitness):
    best = individual[:]
    best[0] = (best[0] + 1) % 2
    for pos in range(1,len(individual)):
        new_individual = individual[:]
        new_individual[pos]= (individual[pos] + 1) % 2
        if fitness(new_individual) > fitness(best):
            best = new_individual
    return best

# Task 2A
# first neighbor
def first_neighbor(individual, fitness):
    """To be DONE."""
    pass

# Task 2B
def random_restart(problem_size, max_iter, restart_times, fitness):
    """To be DONE."""
    pass
    
# Fitness for JB
def evaluate(indiv):
    alfa = 1
    beta = 1.5
    return alfa * sum(indiv) - beta * viola(indiv)


def new_evaluate(alfa,beta):
    def evaluate_indiv(indiv):
        return alfa * sum(indiv) - beta * viola(indiv)
    return evaluate_indiv
	

def viola(indiv):
	# count constraint violations
	comp=len(indiv)
	v=0
	for i in range(1,comp):
		limite= min(i,comp - i - 1)
		vi=0
		for j in range(1,limite+1):
			if (indiv[i]==1) and (indiv[i-j]==1) and (indiv[i+j] == 1):
				vi+=1
		v+=vi
	return v  
    
# Auxiliar
def fenotipo(indiv):
    fen = [i+1 for i in range(len(indiv)) if indiv[i] == 1]
    return fen

    
if __name__ == '__main__':
    matplotlib.use('TkAgg')
    alfa = 1
    beta = 1.5
    eval_new = new_evaluate(alfa,beta)
    sol = jb_hc(1000,100,eval_new)
    res = fenotipo(sol)
    quali = evaluate(sol)
    violations = viola(sol)
    print('INDIV: %s\nQUALIDADE: %s\nVIOLAÇÕES:  %s\nTAMANHO:%s' % (res, quali,violations,len(res)))
