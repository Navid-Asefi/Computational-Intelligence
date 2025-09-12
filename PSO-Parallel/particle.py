import random

# ---- Random vector helpers ----
def random_position(dimension, bounds=(-10, 10)):
    return [random.uniform(bounds[0], bounds[1]) for _ in range(dimension)]
    
def random_velocity(dimension, vmax=2.0):
    return [random.uniform(-vmax, vmax) for _ in range(dimension)]


# ---- Particle Class ----
class Particle:
    def __init__(self, position, velocity, personal_best):
        self.position = position
        self.velocity = velocity
        self.best = personal_best

    @staticmethod
    def swarms(dimension, bounds, population=50):
        """Initialize a swarm of particles"""
        particles = []
        vmax = (bounds[1] - bounds[0]) * 0.1

        for _ in range(population):
            position = random_position(dimension, bounds)
            velocity = random_velocity(dimension, vmax)
            
            particle = Particle(position, velocity, position[:])
            particles.append(particle)

        return particles
