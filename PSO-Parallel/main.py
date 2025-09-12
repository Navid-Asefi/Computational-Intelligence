from optimizer import pso_parallel

# Example objective function (Sphere function)
def sphere(x):
    return sum(xi**2 for xi in x)

if __name__ == "__main__":
    pos, val, idx = pso_parallel(sphere, dimension=50, bounds=(-100,100))
    print(f"Best position {pos} with value {val}, found by particle {idx}")

