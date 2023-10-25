import numpy as np
import multiprocessing
import time

# Define your ODE function
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

def parallel_euler_method(start, end, num_steps, initial_value, num_processes):
    pool = multiprocessing.Pool(processes=num_processes)
    results = []

    step_size = (end - start) / num_steps
    t_values = np.linspace(start, end, num_steps)

    # Split the time values for each process
    t_splits = np.array_split(t_values, num_processes)

    for t_split in t_splits:
        result = pool.apply_async(euler_method, (t_split[0], t_split[-1], len(t_split), initial_value))
        results.append(result)

    pool.close()
    pool.join()

    # Combine the results from all processes
    combined_t_values = []
    combined_y_values = []

    for result in results:
        t_values, y_values = result.get()
        combined_t_values.extend(t_values)
        combined_y_values.extend(y_values)

    return combined_t_values, combined_y_values

if __name__ == '__main__':
    start_value = 0
    end_value = 2
    num_steps = 50000000
    initial_value = 0.5
    num_processes = multiprocessing.cpu_count()

    start = time.time()
    t_values, y_values = parallel_euler_method(start_value, end_value, num_steps, initial_value, num_processes)
    end = time.time()

    execution_time = end - start
    print(f"Execution Time: {execution_time} seconds")

    # You can now plot or analyze the results as needed.

