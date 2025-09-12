from particle import Particle
import random
from multiprocessing import Pool

def pso_parallel(objective_function, dimension, bounds, population=50, iterations=40000,
                 w=0.74, c1=1.42, c2=1.42, num_cores=4):

    # Initialize swarm
    swarm = Particle.swarms(dimension, bounds, population)

    # Helper to evaluate positions in parallel
    def eval_positions(positions):
        with Pool(num_cores) as pool:
            fitnesses = pool.map(objective_function, positions)
        return fitnesses

    # --- Initial global best ---
    positions = [p.position for p in swarm]
    fitnesses = eval_positions(positions)
    
    best_index = fitnesses.index(min(fitnesses))
    global_best = swarm[best_index].position[:]
    best_value = fitnesses[best_index]
    best_particle_index = best_index

    # --- Iteration loop ---
    for _ in range(iterations):

        # Update velocity & position
        for p in swarm:
            new_velocity = []
            for i in range(dimension):
                r1, r2 = random.random(), random.random()
                vel = (w * p.velocity[i] +
                       c1 * r1 * (p.best[i] - p.position[i]) +
                       c2 * r2 * (global_best[i] - p.position[i]))
                new_velocity.append(vel)
            p.velocity = new_velocity
            p.position = [p.position[i] + p.velocity[i] for i in range(dimension)]

        # Evaluate all positions in parallel
        positions = [p.position for p in swarm]
        fitnesses = eval_positions(positions)

        # Update personal bests
        for i, p in enumerate(swarm):
            if fitnesses[i] < objective_function(p.best):
                p.best = p.position[:]

        # Update global best
        best_index = min(range(len(swarm)), key=lambda i: objective_function(swarm[i].best))
        if objective_function(swarm[best_index].best) < best_value:
            global_best = swarm[best_index].best[:]
            best_value = objective_function(global_best)
            best_particle_index = best_index

    return global_best, best_value, best_particle_index
