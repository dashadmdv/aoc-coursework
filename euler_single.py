import numpy as np
import time
import psutil

# ODE function
def f(t, y):
    return y - t**2 + 1

# Euler method for solving ODE
def euler_method(start, end, num_steps, initial_value):
    step_size = (end - start) / num_steps
    t_values = np.linspace(start, end, num_steps)
    y_values = np.zeros(num_steps)
    y_values[0] = initial_value

    for i in range(1, num_steps):
        y_values[i] = y_values[i - 1] + step_size * f(t_values[i - 1], y_values[i - 1])

    return t_values, y_values

if __name__ == '__main__':
    start_value = 0
    end_value = 2
    num_steps = 50000000
    initial_value = 0.5

    # Set CPU affinity to a single core
    psutil.Process().cpu_affinity([0])

    start = time.time()
    t_values, y_values = euler_method(start_value, end_value, num_steps, initial_value)
    end = time.time()

    execution_time = end - start
    print(f"Execution Time: {execution_time} seconds")

