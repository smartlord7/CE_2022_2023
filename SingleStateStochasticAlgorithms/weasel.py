# Task 3: An hill-climbing approach for solving: Me think it is like a weasel
import random
import matplotlib.pyplot as plt

def weasel_hc(problem_size, max_iter, max_dist = 5):
    candidate = random_indiv_char(problem_size)
    print(0,'\t:' ,candidate)
    cost_candi = weasel_fitness(candidate)
    for i in range(max_iter):
        new_neighbor = weasel_neighbor(candidate,max_dist)
        cost_new_neighbor = weasel_fitness(new_neighbor)
        if cost_new_neighbor <= cost_candi: 
            candidate = new_neighbor
            cost_candi = cost_new_neighbor
        print(i+1,'\t:' ,candidate)
    return candidate, cost_candi,i

# define random individuals
def random_indiv_char(size):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '
    return ''.join([random.choice(alphabet) for i in range(size)])

# define the variation operator
def weasel_neighbor(individual, max_dist = 5):
    new_individual = individual[:]
    pos = random.randint(0,len(individual) - 1)
    gene = individual[pos]
    new_gene = aux_neighbor(gene,max_dist)
    new_individual = new_individual[:pos] + new_gene + new_individual[pos+1:]
    return new_individual


def aux_neighbor(char, max_dist):
    distance = random.randint(1,max_dist)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '
    index = alphabet.index(char)
    left_right = random.choice([-1,1])
    new_pos = (index + left_right * distance) % len(alphabet)
    return alphabet[new_pos]

# define fitness function
def weasel_fitness(individual):
    target ='ME THINKS IT IS LIKE A WEASEL'
    return sum([ 1 for i in range(len(target)) if target[i] != individual[i]])
    
def show_data(title,x_axis,y_axis,appearance, data):
    x = range(len(data))
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    
    
    plt.plot(x,data, appearance)
    #plt.savefig('weasel_500a.png')
    plt.show()     
    
if __name__ == '__main__':
    # problem data
    size = 29
    # algorithm configuration
    num_iter = 4000
    max_dist = 5
    # run code
    data = []
    num_runs=50
    for i in range(num_runs):
        candidate,best,n = weasel_hc(size, num_iter, max_dist)
        data.append(best)
    
    show_data('Weasel','Run','Error','r-o',data)        
