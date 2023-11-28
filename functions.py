# create all needed functions in here
import numpy as np


def loss_function_mse(n, i, desired_out,
                      actual_out):  # Just to begin with something doens't have to be this one in the end
    sum = 0
    for i in range(0, i):
        sum += np.power(desired_out - actual_out, 2)
    return 1 / n * sum



