import random

from benchmark import Benchmark


def main():
    # Choose benchmark
    benchmark = Benchmark("Griewank")

    # Get benchmark info
    dims = benchmark.get_dims()
    lb, ub = benchmark.get_bounds()

    # Initialize population
    population = [[random.uniform(lb, ub) for _ in range(dims)] for _ in range(50)]

    # Evaluate fitness
    fitness = [benchmark.evaluate(ind) for ind in population]

    print("First chromosome:", population[0])
    print("Fitness:", fitness[0])


main()
