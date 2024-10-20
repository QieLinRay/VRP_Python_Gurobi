import numpy as np
from data import Data
from model import CVRP

"""
10.20 modification:

1. use numpy to generate data rather than "for" loop for fast speed and simplification.
2. Implement a Data class wrap generated data to avoid conflicts with global variables.
3. Implement a CVRP class that accepts data as input and provides a solve() method to execute the solution process.
"""


def main(k, n, q, s):

    np.random.seed(s)
    data = Data(K=k, N=n, Q=q)
    model = CVRP(data)

    model.solve()


if __name__ == '__main__':

    """define hyperparameter"""
    K = 5  # the number of the vehicles
    N = 4  # the number of customers
    Q = 50  # the capacity of the vehicles
    seed = 1

    main(K, N, Q, seed)



