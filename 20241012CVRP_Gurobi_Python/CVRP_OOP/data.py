import random
import math
import numpy as np


class Data:
    def __init__(self, K: int, N: int, Q: int, M: int = 99999):
        self.t_nn1 = None
        self.y_n = None
        self.x_n = None
        self.q_n = None
        self.__is_initialized = False

        self.K = K
        self.N = N
        self.Q = Q
        self.M = M

        self.initialize()

    def initialize(self):

        if not self.__is_initialized:
            """use numpy to accelerate and simplify the data generation process"""

            self.q_n = np.random.randint(10, 15, self.N)  # the demand of customers [low, high, size]
            self.x_n = np.random.randint(0, 50, self.N + 2)
            self.y_n = np.random.randint(0, 50, self.N + 2)
            # the travel time from n to n1
            self.t_nn1 = (np.abs(np.expand_dims(self.x_n, axis=1) - np.expand_dims(self.x_n, axis=0)) +
                          np.abs(np.expand_dims(self.y_n, axis=1) - np.expand_dims(self.y_n, axis=0)))

            self.__is_initialized = True
            print("Data initialized!")
        else:
            print("Data already initialized!")

