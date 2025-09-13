from benchmarkfcns import (
    bartelsconn,
    beale,
    booth,
    brent,
    brown,
    crossintray,
    griewank,
    matyas,
    periodic,
    powellsum,
    qing,
    quartic,
    rastrigin,
    rosenbrock,
    salomon,
    sphere,
    styblinskitank,
    threehumpcamel,
)


class Benchmark:
    functions = {
        "Sphere": {"function": sphere, "bounds": (-100, 100), "default_dim": 50},
        "Quartic Noise": {
            "function": quartic,
            "bounds": (-1.28, 1.28),
            "default_dim": 50,
        },
        "Powell Sum": {"function": powellsum, "bounds": (-1, 1), "default_dim": 50},
        "Rosenbrock": {"function": rosenbrock, "bounds": (-30, 30), "default_dim": 50},
        "Brown": {"function": brown, "bounds": (-1, 4), "default_dim": 50},
        "Three-Hump Camel": {
            "function": threehumpcamel,
            "bounds": (-5, 5),
            "default_dim": 2,
        },
        "Beale": {"function": beale, "bounds": (-4.5, 4.5), "default_dim": 2},
        "Booth": {"function": booth, "bounds": (-10, 10), "default_dim": 2},
        "Brent": {"function": brent, "bounds": (-10, 10), "default_dim": 2},
        "Matyas": {"function": matyas, "bounds": (-10, 10), "default_dim": 2},
        "Rastrigin": {
            "function": rastrigin,
            "bounds": (-5.12, 5.12),
            "default_dim": 50,
        },
        "Periodic": {"function": periodic, "bounds": (-10, 10), "default_dim": 50},
        "Qing": {"function": qing, "bounds": (-500, 500), "default_dim": 50},
        "Salomon": {"function": salomon, "bounds": (-100, 100), "default_dim": 50},
        "Styblinski-Tang": {
            "function": styblinskitank,
            "bounds": (-5, 5),
            "default_dim": 50,
        },
        "Griewank": {"function": griewank, "bounds": (-100, 100), "default_dim": 50},
        "Cross-in-tray": {
            "function": crossintray,
            "bounds": (-10, 10),
            "default_dim": 2,
        },
        "Bartels Conn": {
            "function": bartelsconn,
            "bounds": (-500, 500),
            "default_dim": 2,
        },
    }

    def __init__(self, name: str):
        if name not in self.functions:
            raise ValueError(f"Unknown benchmark function: {name}")

        self.name = name
        self.func = self.functions[name]["function"]
        self.bounds = self.functions[name]["bounds"]
        self.dims = self.functions[name]["default_dim"]

    def evaluate(self, chromosome):
        return self.func(chromosome)

    def get_bounds(self):
        return self.bounds

    def get_dims(self):
        return self.dims
