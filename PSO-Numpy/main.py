import numpy as np
from optimizer import pso_numpy

# ------------------- Benchmark Functions ------------------- #

def sphere(x):
    return np.sum(x**2)

def quartic_noise(x):
    n = len(x)
    return np.sum([(i+1) * (x[i]**4) for i in range(n)]) + np.random.rand()

def powell_sum(x):
    return np.sum([abs(x[i])**(i+2) for i in range(len(x))])

def schwefel_220(x):
    return np.sum(np.abs(x))

def schwefel_221(x):
    return np.max(np.abs(x))

def step(x):
    return np.sum((np.floor(x + 0.5))**2)

def stepint(x):
    return np.sum((np.floor(x))**2)

def schwefel_120(x):
    total = 0
    for i in range(len(x)):
        total += (np.sum(x[:i+1]))**2
    return total

def schwefel_222(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def schwefel_223(x):
    return np.sum(x**10)

def rosenbrock(x):
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (x[:-1]-1)**2)

def brown(x):
    return np.sum((x[:-1]**2)**(x[1:]**2+1) + (x[1:]**2)**(x[:-1]**2+1))

def dixon_price(x):
    return (x[0]-1)**2 + np.sum([(i+1)*(2*x[i]**2 - x[i-1])**2 for i in range(1, len(x))])

def powell_singular(x):
    n = len(x)//4
    total = 0
    for i in range(n):
        xi1, xi2, xi3, xi4 = x[4*i:4*i+4]
        total += (xi1 + 10*xi2)**2 + 5*(xi3 - xi4)**2 + (xi2-2*xi3)**4 + 10*(xi1-xi4)**4
    return total

def zakharov(x):
    i = np.arange(1, len(x)+1)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5*i*x)
    return sum1 + sum2**2 + sum2**4

def xin_she_yang(x):
    return np.sum(np.abs(x)) * np.exp(-np.sum(np.sin(x**2)))

def perm_od_beta(x, beta=10):
    n = len(x)
    outer = 0
    for i in range(1, n+1):
        inner = 0
        for j in range(1, n+1):
            inner += (j**i + beta) * ((x[j-1]/j)**i - 1)
        outer += inner**2
    return outer

def three_hump_camel(x):
    x1, x2 = x
    return 2*x1**2 - 1.05*x1**4 + (x1**6)/6 + x1*x2 + x2**2

def beale(x):
    x1, x2 = x
    return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*(x2**2))**2 + (2.625 - x1 + x1*(x2**3))**2

def booth(x):
    x1, x2 = x
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

def brent(x):
    x1, x2 = x
    return (x1+10)**2 + (x2+10)**2 + np.exp(-(x1**2 + x2**2))

def matyas(x):
    x1, x2 = x
    return 0.26*(x1**2 + x2**2) - 0.48*x1*x2

def schaffer_n4(x):
    x1, x2 = x
    num = np.cos(np.sin(abs(x1**2 - x2**2)))**2 - 0.5
    den = (1+0.001*(x1**2+x2**2))**2
    return 0.5 + num/den

def wayburn_seader3(x):
    x1, x2 = x
    return ((x1**6 + x2**4 - 17)**2 + (2*x1 + x2 - 4)**2)

def leon(x):
    x1, x2 = x
    return 100*(x2 - x1**2)**2 + (1-x1)**2

def schwefel_226(x):
    return 418.9829*len(x) - np.sum(x*np.sin(np.sqrt(np.abs(x))))

def rastrigin(x):
    return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))

def periodic(x):
    return 1 + np.sum(np.sin(x)**2) - 0.1*np.exp(-np.sum(x**2))

def qing(x):
    return np.sum((x**2 - np.arange(1,len(x)+1))**2)

def alpine1(x):
    return np.sum(np.abs(x*np.sin(x) + 0.1*x))

def xin_she_yang2(x):
    return np.sum(np.abs(x)) * np.exp(-np.sum(np.sin(x**2)))

def ackley(x):
    n = len(x)
    return -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/n)) - np.exp(np.sum(np.cos(2*np.pi*x))/n) + 20 + np.e

def trignometric2(x):
    n = len(x)
    return np.sum((n - np.sum(np.cos(x) + i*(1-np.cos(x[i])) for i in range(n)))**2)

def salomon(x):
    normx = np.sqrt(np.sum(x**2))
    return 1 - np.cos(2*np.pi*normx) + 0.1*normx

def styblinski_tang(x):
    return 0.5*np.sum(x**4 - 16*x**2 + 5*x)

def griewank(x):
    sum_part = np.sum(x**2)/4000
    prod_part = np.prod(np.cos(x/np.sqrt(np.arange(1,len(x)+1))))
    return sum_part - prod_part + 1

def xin_she_yang4(x):
    return np.sum(np.random.rand(len(x)) * np.abs(x)**i for i in range(1, len(x)+1))

def xin_she_yangN2(x):
    return np.sum(np.sin(x)**2) - np.exp(-np.sum(x**2))

def penalized(x):
    y = 1 + (x+1)/4
    term1 = np.pi/len(x) * (10*np.sin(np.pi*y[0])**2 + np.sum((y[:-1]-1)**2 * (1+10*np.sin(np.pi*y[1:])**2)) + (y[-1]-1)**2)
    u = np.sum((x>10)*(x-10)**2 + (x<-10)*(-10-x)**2)
    return term1 + u

def egg_crate(x):
    x1, x2 = x
    return x1**2 + x2**2 + 25*(np.sin(x1)**2 + np.sin(x2)**2)

def ackley_n3(x):
    x1, x2 = x
    return -200*np.exp(-0.02*np.sqrt(x1**2+x2**2)) + 5*np.exp(np.cos(3*x1)+np.sin(3*x2))

def adjiman(x):
    x1, x2 = x
    return np.cos(x1)*np.sin(x2) - x1/(x2**2+1)

def bird(x):
    x1, x2 = x
    return np.sin(x1)*np.exp((1-np.cos(x2))**2) + np.cos(x2)*np.exp((1-np.sin(x1))**2) + (x1-x2)**2

def camel_six_hump(x):
    x1, x2 = x
    return (4 - 2.1*x1**2 + (x1**4)/3)*x1**2 + x1*x2 + (-4+4*x2**2)*x2**2

def branin_rcos(x):
    x1, x2 = x
    a, b, c, r, s, t = 1, 5.1/(4*np.pi**2), 5/np.pi, 6, 10, 1/(8*np.pi)
    return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s

def hartman3(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3,10,30],[0.1,10,35],[3,10,30],[0.1,10,35]])
    P = 1e-4*np.array([[3689,1170,2673],[4699,4387,7470],[1091,8732,5547],[381,5743,8828]])
    outer = 0
    for i in range(4):
        inner = np.sum(A[i]*(x-P[i])**2)
        outer += alpha[i]*np.exp(-inner)
    return -outer

def hartman6(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],[3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]])
    P = 1e-4*np.array([[1312,1696,5569,124,8283,5886],[2329,4135,8307,3736,1004,9991],[2348,1451,3522,2883,3047,6650],[4047,8828,8732,5743,1091,381]])
    outer = 0
    for i in range(4):
        inner = np.sum(A[i]*(x-P[i])**2)
        outer += alpha[i]*np.exp(-inner)
    return -outer

def cross_in_tray(x):
    x1, x2 = x
    fact = np.exp(abs(100 - np.sqrt(x1**2+x2**2)/np.pi))
    return -0.0001*(abs(np.sin(x1)*np.sin(x2)*fact)+1)**0.1

def bartels_conn(x):
    x1, x2 = x
    return abs(x1**2+x2**2+x1*x2) + abs(np.sin(x1)) + abs(np.cos(x2))


def run_pso_experiment(func, bounds, dimension, runs=20):
    results = []
    for _ in range(runs):
        _, best_val = pso_numpy(func, dimension=dimension, bounds=bounds)
        results.append(best_val)
    results = np.array(results)
    return {
        "mean": np.mean(results),
        "min": np.min(results),
        "std": np.std(results)
    }

# ------------------- Example Usage ------------------- #
if __name__ == "__main__":
    stats = run_pso_experiment(sphere, bounds=(-100, 100), dimension=50, runs=20)
    print("Stats:", stats)
