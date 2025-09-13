import numpy as np

# ---- Particle Class ----
class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best = position.copy()

    @staticmethod
    def swarms(dimension, bounds, population=50):
        positions = np.random.uniform(bounds[0], bounds[1], (population, dimension))
        vmax = (bounds[1] - bounds[0]) * 0.1
        velocities = np.random.uniform(-vmax, vmax, (population, dimension))
        particles = [Particle(positions[i], velocities[i]) for i in range(population)]
        return particles
