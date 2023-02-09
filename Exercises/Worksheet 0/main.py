from math import *
import matplotlib
import matplotlib.pyplot as plt


def range_(begin: float,
           end: float,
           step: float = 1.0) -> list:
    r = list()
    dec_places = str(step)[::-1].find('.')

    while begin <= end:
        begin = round(begin, dec_places)
        r.append(begin)

        begin += step

    return r


def dim1_derivative(function: str,
                    val: float,
                    precision: float = 0.000000001,
                    var_name: str = "x",
                    allowed_func: dict = None) -> float:
    default_allowed_func = {"sin": sin,
                            "cos": cos,
                            "tan": tan,
                            "sinh": sinh,
                            "cosh": cosh,
                            "tanh": tanh,
                            "log2": log2}

    if allowed_func is None:
        allowed_func = default_allowed_func
    else:
        allowed_func.update(default_allowed_func)

    return (eval(function, allowed_func, {var_name: val + precision}) - eval(function, allowed_func,
                                                                             {var_name: val})) / precision


def permutations(s: set) -> list:
    p = list()

    def permutations_(s2: set, permutation: list, idx: int):
        if idx == len(s):
            p.append(permutation)
        else:
            for elem in s2:
                s3 = s2.copy()
                s3.remove(elem)
                permutation2 = permutation.copy()
                permutation2.append(elem)

                permutations_(s3, permutation2, idx + 1)

    permutations_(s, list(), 0)

    return p


def sum_sparse_vec(sparse_vec1: dict, sparse_vec2: dict) -> dict:
    result = dict()

    for k in sparse_vec1.keys():
        if k in sparse_vec2:
            result[k] = sparse_vec1[k] + sparse_vec2[k]
        else:
            result[k] = sparse_vec1[k]

    for k in sparse_vec2.keys():
        if k not in result:
            result[k] = sparse_vec2[k]

    return result


def outer_product_sparse_vec(sparse_vec1: dict, sparse_vec2: dict) -> dict:
    result = dict()

    for k in sparse_vec1.keys():
        if k in sparse_vec2:
            result[k] = sparse_vec1[k] * sparse_vec2[k]
        else:
            result[k] = sparse_vec1[k]

    for k in sparse_vec2.keys():
        if k not in result:
            result[k] = sparse_vec2[k]

    return result


def dot_product_sparse_vec(sparse_vec1: dict, sparse_vec2: dict) -> float:
    outer_prod = outer_product_sparse_vec(sparse_vec1, sparse_vec2)
    result = float()

    for k in outer_prod.keys():
        result += outer_prod[k]

    return result


def main() -> None:
    matplotlib.use('TkAgg')

    # P0.1 Float range generator function
    # print(range_(1.6, 10, 0.4))

    # P0.2 1D derivative
    # print(dim1_derivative("log10(x)", 5, allowed_func={"log10": log10}))
    # f = "x**3 - 2*x**2 - 11*x + 12"
    # f2 = "3*x**2 - 4*x - 11"
    # f3 = "6*x - 4"
    # print(dim1_derivative(f, 2))
    # print(dim1_derivative(f2, 2))

    # P0.3 Plot of the 1st and 2nd derivative of a function
    # x = range_(0.1, 100, 0.1)
    # y = [log2(v) for v in x]
    # y2 = [dim1_derivative("log2(x)", v) for v in x]
    # plt.plot(x, y)
    # plt.plot(x, y2)
    # plt.show()

    # P0.4 Permutations of a given set
    # print(permutations({1, 2, 3, 4}))

    # sv1 = {2:5, 6:8, 8:32}
    # sv2 = {2:1, 5:2, 8:10}

    # P0.5 Sum of two sparse vectors
    # print(sum_sparse_vec(sv1, sv2))

    # P0.6 Outer product of two sparse vectors
    # print(outer_product_sparse_vec(sv1, sv2))

    # P0.7 Dot (intern) product of two sparse vectors
    # print(dot_product_sparse_vec(sv1, sv2))

    # sm1 = {(1, 2): 5, (3, 3): 2, (2, 2): 10}
    # sm2 = {(1, 1): 2, (3, 3): 7, (2, 2): 52}

    # P0.8 Sum of two sparse matrices (the code for sparse vectors is compatible)
    # print(sum_sparse_vec(sm1, sm2))

    # Outer product of two sparse matrices
    # print(outer_product_sparse_vec(sm1, sm2))

    # P0.9 Dot (intern) product of two sparse matrices (the code for sparse vectors is compatible)
    # print(dot_product_sparse_vec(sm1, sm2))

    pass


if __name__ == '__main__':
    main()
