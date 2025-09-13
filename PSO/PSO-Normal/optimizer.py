from particle import Particle
import random

def pso(objective_function, dimension, bounds, population=50, iterations=4000,
        w=0.74, c1=1.42, c2=1.42):

    # Initialize swarm
    swarm = Particle.swarms(dimension, bounds, population)
    
    # Find initial global best
    best_particle = swarm[0]
    best_index = 0
    for i, p in enumerate(swarm):
        if objective_function(p.position) < objective_function(best_particle.position):
            best_particle = p
            best_index = i

    global_best = best_particle.position[:]
    best_value = objective_function(global_best)
    best_particle_index = best_index 


    # Iteration loop
    for _ in range(iterations):

        for p in swarm:
            # --- Update velocity ---
            new_velocity = []

            for i in range(dimension):
                r1, r2 = random.random(), random.random()
                vel = (w * p.velocity[i] +
                       c1 * r1 * (p.best[i] - p.position[i]) +
                       c2 * r2 * (global_best[i] - p.position[i]))
                
                new_velocity.append(vel)

            p.velocity = new_velocity

            # --- Update position ---
            p.position = [p.position[i] + p.velocity[i] for i in range(dimension)]

            # --- Update personal best ---
            if objective_function(p.position) < objective_function(p.best):
                p.best = p.position[:]

        # --- Update global best ---
        best_particle = swarm[0]
        best_index = 0
        for i, p in enumerate(swarm):
            if objective_function(p.best) < objective_function(best_particle.best):
                best_particle = p
                best_index = i

        if objective_function(best_particle.best) < objective_function(global_best):
            global_best = best_particle.best[:]
            best_value = objective_function(global_best)
            best_particle_index = best_index 

    return global_best, best_value, best_particle_index


