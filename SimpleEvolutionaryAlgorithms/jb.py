"""
jb.py
The code below is given without any warranty!
Ernesto Costa, February, 2023
"""
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from sea_bin_students import *


# from utils import *

# João Brandão
from stat_alunos import histogram


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


def run_exp(crossover_func, log_file_name):
    generations = 250
    pop_size = 100
    cromo_size = 50
    prob_muta = 0.01
    prob_cross = 0.9
    tour_size = 3
    elite_percent = 0.02
    n_runs = 30

    best_all = [0 for _ in range(generations)]
    avg_best = [0 for _ in range(generations)]
    with open(log_file_name, 'w') as f:
        for i in range(1, n_runs + 1):
            avg_time, best_time, best = sea(generations, pop_size, cromo_size, prob_muta, prob_cross,
                                            tour_sel(tour_size),
                                            crossover_func, muta_bin,
                                            sel_survivors_elite(elite_percent), fitness)
            for i in range(generations):
                avg_best[i] += avg_time[i]
                curr = best_time[i]
                if curr > best_all[i]:
                    best_all[i] = curr

            numb_viola = viola(phenotype(best), cromo_size)
            print('Violations: ', numb_viola)
            display(best, phenotype)
            # display_stat_1(best_time, avg_time)
            f.write(str(best[1]) + '\n')

    for i in range(generations):
        avg_best[i] = avg_best[i] / n_runs

    display_stat_n(best_all, avg_best)

    data = []
    with open(log_file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(float(line))
    histogram(data, 'Fitness histogram', 'Occurrences', 'Fitness')

    return data


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    # GENERATIONS = [i for i in range(10, 300, 10)]
    # SIZE_POPULATION = [i for i in range(100, 1000, 100)]
    # SIZE_CROMO = [i for i in range(20, 100, 20)]
    # PROB_MUTATION = [i for i in np.arange(0.01, 1.0, 0.02)]
    # PROB_CROSSOVER = [i for i in np.arange(0.5, 1.0, 0.05)]
    # PERCENT_TOURNAMENT = [i for i in np.arange(0.02, 1.0, 0.05)]
    # PERCENT_ELITE = [i for i in np.arange(0.02, 1.0, 0.02)]

    data1 = run_exp(one_point_cross, 'one_point_cross.txt')
    data2 = run_exp(two_points_cross, 'two_points_cross.txt')
    data3 = run_exp(uniform_cross, 'uniform_cross.txt')

    plt.figure()
    data = pd.DataFrame({"One point": data1, "Two points": data2, "Uniform" : data3})
    ax = data[['One point', 'Two points', 'Uniform']].plot(kind='box', title='boxplot')
    plt.show()
