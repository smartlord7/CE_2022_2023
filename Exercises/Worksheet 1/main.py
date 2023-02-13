import math
import random
import time as t

import matplotlib
import matplotlib.pyplot as plt


def derivative(f, delta_x=0.000001):
    """Return the derivative of a function"""
    def der(x):
        return (f(x + delta_x) -f(x)) / delta_x
    
    return der


def display_data(data):
    x = list(range(len(data)))
    plt.grid(True)
    plt.axhline(c='black')
    plt.axvline(c='black')
    plt.plot(x,data, 'r')
    plt.show()
    
    
def display_function(f, x_min, x_max, delta=0.1):
    x = list(frange(x_min, x_max,delta))
    y = [f(i) for i in x]
    plt.title(f.__name__)
    plt.grid(True)
    plt.axhline(c='black')
    plt.axvline(c='black')
    plt.xlabel('X')
    plt.ylabel('Y= '+f.__name__ + '(X)')
    plt.plot(x, y, 'r')
    plt.show()


def frange(n1,n2=None,n3=1.0):
    """
    Range with floats.
    Can be called as with range:
    frange(n)
    frange(n1,n2)
    fange(n1,n2,n3)
    """
    if n2 == None:
        n2 = n1
        n1 = 0.0
    nextn = n1

    while (n3 >= 0.0 and nextn <= n2) or (n3 < 0.0 and nextn >= n2):
        yield nextn
        nextn += n3


def gradient_ascent(function,
                    domain,
                    epsilon=0.001,
                    learning_rate=1.0,
                    time_limit=10):

    begin = t.perf_counter()
    best = random.uniform(domain[0], domain[1])
    der = derivative(function)

    elapsed = int()
    gradients = list()
    gradient = der(best)
    gradients.append(gradient)

    while abs(gradient) >= epsilon:
        best += learning_rate * gradient
        gradient = der(best)
        gradients.append(gradient)

        elapsed = t.perf_counter() - begin
        if elapsed >= time_limit:
            break

    return best, gradients, elapsed


def rdm_restart_gradient_ascent(function,
                    domain,
                    epsilon=0.001,
                    learning_rate=0.1,
                    time_limit=10):

    begin = t.perf_counter()
    x = random.uniform(domain[0], domain[1])
    best = x
    der = derivative(function)
    i = int()

    while t.perf_counter() - begin <= time_limit:
        gradients = list()
        gradient = der(x)
        gradients.append(gradient)

        while abs(gradient) >= epsilon:
            x += learning_rate * gradient
            gradient = der(x)
            gradients.append(gradient)

        if function(x) >= function(best):
            best = x

        x = random.uniform(domain[0], domain[1])
        print("Iteration %d: best: %s" % (i, best))
        i += 1

    return best


def newton_method(function,
                    domain,
                    epsilon=0.001,
                    learning_rate=1.0,
                    time_limit=10):

    begin = t.perf_counter()
    best = random.uniform(domain[0], domain[1])
    der = derivative(function)
    der2 = derivative(der)

    elapsed = int()
    gradients = list()
    hessians = list()
    gradient = der(best)
    hessian = der2(best)
    gradients.append(gradient)

    while abs(gradient) >= epsilon:
        best -= learning_rate * (hessian ** -1) * gradient
        gradient = der(best)
        hessian = der2(best)
        gradients.append(gradient)
        hessians.append(hessian)

        elapsed = t.perf_counter() - begin
        if elapsed >= time_limit:
            break

    return best, gradients, hessians, elapsed


def f1(x):
    return x**3 - 2*x + 2


def f2(x):
    return math.sin(x) / x


def f3(x):
    return 2 * math.sin(x) + math.sin(2 * x)


def main():
    matplotlib.use('TkAgg')
    functions = [f1, f2, f3]
    x_min = -100
    x_max = 100
    domain = [x_min, x_max]

    i = 1
    for f in functions:
        print("Function %d" % i)
        plt.figure()
        display_function(f, x_min, x_max)
        plt.figure()
        best1, gradients, elapsed = gradient_ascent(f, domain)
        print("     Gradient ascent\n"
              "        Best: x = %s, f(x) = %s\n"
              "        Converged after: %s s" % (str(best1),
                                                 str(f(best1)),
                                                 str(elapsed)))
        plt.title("Gradient evolution")
        display_data(gradients)
        best2, gradients2, hessians, elapsed2 = newton_method(f, domain)
        print("     Newton method\n"
              "        Best: x = %s, f(x) = %s\n"
              "        Converged after: %s s" % (str(best2),
                                                 str(f(best2)),
                                                 str(elapsed2)))
        plt.title("Gradient evolution")
        display_data(gradients2)
        plt.title("Hessian evolution")
        display_data(hessians)
        #best = rdm_restart_gradient_ascent(f, domain)
        i += 1

    pass


if __name__ == '__main__':
    main()

