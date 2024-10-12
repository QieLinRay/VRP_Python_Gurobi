import random
import math

K = 5       # the number of the vehicles
N = 4       # the number of customers
Q = 50      # the capacity of the vehicles
q_n = [0 for n in range(N)]     # the demand of customers
x_n =[0 for n in range(N + 2)]
y_n = [0 for n in range(N + 2)]

for n in range(N + 2):
    x_n[n] = random.randint(0,50)
    y_n[n] = random.randint(0,50)

t_nn1 = [[] for n in range(N + 2)]      # the travel time from n to n1
for n in range(N + 2):
    t_nn1[n] = [[] for n1 in range(N + 2)]
    for n1 in range(N + 2):
        t_nn1[n][n1] = abs(x_n[n] - x_n[n1]) + abs(y_n[n] - y_n[n1])

for n in range(N):
    q_n[n] = random.randint(10,15)

M = 99999999