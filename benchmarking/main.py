import random

import numpy as np
from benchmark import Benchmark


def run_single(name, pop_size=50):
    """Run a single benchmark on its own dimensionality and return fitness stats"""
    benchmark = Benchmark(name)
    dims = benchmark.get_dims()
    lb, ub = benchmark.get_bounds()

    # Generate population in the correct dimension
    population = [
        [random.uniform(lb, ub) for _ in range(dims)] for _ in range(pop_size)
    ]

    fitness = []
    for ind in population:
        val = benchmark.evaluate(ind)
        # Ensure scalar value
        if isinstance(val, np.ndarray):
            val = val.item() if val.size == 1 else float(val.sum())
        fitness.append(val)

    mean_fit = float(np.mean(fitness))
    std_fit = float(np.std(fitness, ddof=1)) if len(fitness) > 1 else 0.0
    return mean_fit, std_fit


def main():
    available = list(Benchmark.functions.keys())
    print("Available benchmark functions:")
    for i, name in enumerate(available, 1):
        print(f"{i}. {name}")
    print(f"{len(available) + 1}. Run ALL")

    choice = input("Select a function by number, name, or 'all': ").strip()

    if choice.lower() in {"all", str(len(available) + 1)}:
        results = []
        for name in available:
            mean_fit, std_fit = run_single(name)
            results.append((name, mean_fit, std_fit))

        # Rank by mean fitness (lower is better)
        results.sort(key=lambda x: x[1])

        print("\n=== Benchmark Results ===")
        print(f"{'Rank':<5}{'Function':<20}{'Mean':<20}{'StdDev':<20}")
        for rank, (name, mean_fit, std_fit) in enumerate(results, 1):
            print(f"{rank:<5}{name:<20}{mean_fit:<20.6f}{std_fit:<20.6f}")

    else:
        if choice.isdigit():
            idx = int(choice) - 1
            if idx < 0 or idx >= len(available):
                raise ValueError("Invalid index")
            chosen = available[idx]
        else:
            if choice not in available:
                raise ValueError("Invalid name")
            chosen = choice

        print(f"\nYou selected: {chosen}\n")
        mean_fit, std_fit = run_single(chosen)
        print(f"Mean fitness: {mean_fit:.6f}")
        print(f"StdDev fitness: {std_fit:.6f}")


if __name__ == "__main__":
    main()
