import numpy as np
from particle import Particle


def pso_numpy(
    objective_function,
    dimension,
    bounds,
    population=50,
    iterations=40000,
    w=0.74,
    c1=1.42,
    c2=1.42,
):
    swarm = Particle.swarms(dimension, bounds, population)

    # Convert positions and bests to arrays for vector operations
    positions = np.array([p.position for p in swarm])
    velocities = np.array([p.velocity for p in swarm])
    personal_bests = positions.copy()

    # Initial global best
    fitness = np.array([objective_function(pos) for pos in positions])
    best_idx = np.argmin(fitness)
    global_best = positions[best_idx].copy()
    best_value = fitness[best_idx]

    for _ in range(iterations):
        r1 = np.random.rand(population, dimension)
        r2 = np.random.rand(population, dimension)

        # Update velocity
        velocities = (
            w * velocities
            + c1 * r1 * (personal_bests - positions)
            + c2 * r2 * (global_best - positions)
        )

        # Update position
        positions += velocities

        # Evaluate fitness
        fitness = np.array([objective_function(pos) for pos in positions])

        # Update personal bests
        improved = fitness < np.array([objective_function(p.best) for p in swarm])
        for i, p in enumerate(swarm):
            if improved[i]:
                p.best = positions[i].copy()
                personal_bests[i] = positions[i].copy()

        # Update global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_value:
            global_best = positions[best_idx].copy()
            best_value = fitness[best_idx]

    return global_best, best_value
