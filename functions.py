# create all needed functions in here
import numpy as np

# desired_out & actual_out would have to be np arrays
def loss_function_mse(desired_out,
                      actual_out):  # Just to begin with something doens't have to be this one in the end
    return np.mean(np.power(desired_out - actual_out, 2))

def mse_prime(desired_out,
                actual_out):
    return 2*(actual_out-desired_out)/desired_out.size;



