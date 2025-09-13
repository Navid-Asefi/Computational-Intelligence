import markdown
import numpy as np
from optimizer import pso_numpy
from weasyprint import HTML


def sphere(x):
    return np.sum(x**2)


def quartic_noise(x):
    n = len(x)
    return np.sum([(i + 1) * (x[i] ** 4) for i in range(n)]) + np.random.rand()


def powell_sum(x):
    return np.sum([abs(x[i]) ** (i + 2) for i in range(len(x))])


def schwefel_220(x):
    return np.sum(np.abs(x))


def schwefel_221(x):
    return np.max(np.abs(x))


def step(x):
    return np.sum((np.floor(x + 0.5)) ** 2)


def stepint(x):
    return np.sum((np.floor(x)) ** 2)


def schwefel_120(x):
    total = 0
    for i in range(len(x)):
        total += (np.sum(x[: i + 1])) ** 2
    return total


def schwefel_222(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))


def schwefel_223(x):
    return np.sum(x**10)


def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)


def brown(x):
    return np.sum((x[:-1] ** 2) ** (x[1:] ** 2 + 1) + (x[1:] ** 2) ** (x[:-1] ** 2 + 1))


def dixon_price(x):
    return (x[0] - 1) ** 2 + np.sum(
        [(i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2 for i in range(1, len(x))]
    )


def powell_singular(x):
    n = len(x) // 4
    total = 0
    for i in range(n):
        xi1, xi2, xi3, xi4 = x[4 * i : 4 * i + 4]
        total += (
            (xi1 + 10 * xi2) ** 2
            + 5 * (xi3 - xi4) ** 2
            + (xi2 - 2 * xi3) ** 4
            + 10 * (xi1 - xi4) ** 4
        )
    return total


def zakharov(x):
    i = np.arange(1, len(x) + 1)
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * i * x)
    return sum1 + sum2**2 + sum2**4


def xin_she_yang(x):
    return np.sum(np.abs(x)) * np.exp(-np.sum(np.sin(x**2)))


def perm_od_beta(x, beta=10):
    n = len(x)
    outer = 0
    for i in range(1, n + 1):
        inner = 0
        for j in range(1, n + 1):
            inner += (j**i + beta) * ((x[j - 1] / j) ** i - 1)
        outer += inner**2
    return outer


def three_hump_camel(x):
    x1, x2 = x
    return 2 * x1**2 - 1.05 * x1**4 + (x1**6) / 6 + x1 * x2 + x2**2


def beale(x):
    x1, x2 = x
    return (
        (1.5 - x1 + x1 * x2) ** 2
        + (2.25 - x1 + x1 * (x2**2)) ** 2
        + (2.625 - x1 + x1 * (x2**3)) ** 2
    )


def booth(x):
    x1, x2 = x
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def brent(x):
    x1, x2 = x
    return (x1 + 10) ** 2 + (x2 + 10) ** 2 + np.exp(-(x1**2 + x2**2))


def matyas(x):
    x1, x2 = x
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


def schaffer_n4(x):
    x1, x2 = x
    num = np.cos(np.sin(abs(x1**2 - x2**2))) ** 2 - 0.5
    den = (1 + 0.001 * (x1**2 + x2**2)) ** 2
    return 0.5 + num / den


def wayburn_seader3(x):
    x1, x2 = x
    return (x1**6 + x2**4 - 17) ** 2 + (2 * x1 + x2 - 4) ** 2


def leon(x):
    x1, x2 = x
    return 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2


def schwefel_226(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def periodic(x):
    return 1 + np.sum(np.sin(x) ** 2) - 0.1 * np.exp(-np.sum(x**2))


def qing(x):
    return np.sum((x**2 - np.arange(1, len(x) + 1)) ** 2)


def alpine1(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x))


def xin_she_yang2(x):
    return np.sum(np.abs(x)) * np.exp(-np.sum(np.sin(x**2)))


def ackley(x):
    n = len(x)
    return (
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
        + 20
        + np.e
    )


def trignometric2(x):
    n = len(x)
    return np.sum(
        [(n - np.sum(np.cos(x)) + i * (1 - np.cos(x[i]))) ** 2 for i in range(n)]
    )


def salomon(x):
    normx = np.sqrt(np.sum(x**2))
    return 1 - np.cos(2 * np.pi * normx) + 0.1 * normx


def styblinski_tang(x):
    return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)


def griewank(x):
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_part - prod_part + 1


def xin_she_yang4(x):
    return np.sum([np.random.rand() * np.abs(x[i]) ** (i + 1) for i in range(len(x))])


def xin_she_yangN2(x):
    return np.sum(np.sin(x) ** 2) - np.exp(-np.sum(x**2))


def penalized(x):
    y = 1 + (x + 1) / 4
    term1 = (
        np.pi
        / len(x)
        * (
            10 * np.sin(np.pi * y[0]) ** 2
            + np.sum((y[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[1:]) ** 2))
            + (y[-1] - 1) ** 2
        )
    )
    u = np.sum((x > 10) * (x - 10) ** 2 + (x < -10) * (-10 - x) ** 2)
    return term1 + u


def egg_crate(x):
    x1, x2 = x
    return x1**2 + x2**2 + 25 * (np.sin(x1) ** 2 + np.sin(x2) ** 2)


def ackley_n3(x):
    x1, x2 = x
    return -200 * np.exp(-0.02 * np.sqrt(x1**2 + x2**2)) + 5 * np.exp(
        np.cos(3 * x1) + np.sin(3 * x2)
    )


def adjiman(x):
    x1, x2 = x
    return np.cos(x1) * np.sin(x2) - x1 / (x2**2 + 1)


def bird(x):
    x1, x2 = x
    return (
        np.sin(x1) * np.exp((1 - np.cos(x2)) ** 2)
        + np.cos(x2) * np.exp((1 - np.sin(x1)) ** 2)
        + (x1 - x2) ** 2
    )


def camel_six_hump(x):
    x1, x2 = x
    return (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2


def branin_rcos(x):
    x1, x2 = x
    a, b, c, r, s, t = 1, 5.1 / (4 * np.pi**2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def hartman3(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    P = 1e-4 * np.array(
        [[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]]
    )
    outer = 0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i]) ** 2)
        outer += alpha[i] * np.exp(-inner)
    return -outer


def hartman6(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    outer = 0
    for i in range(4):
        inner = np.sum(A[i] * (x - P[i]) ** 2)
        outer += alpha[i] * np.exp(-inner)
    return -outer


def cross_in_tray(x):
    x1, x2 = x
    fact = np.exp(abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))
    return -0.0001 * (abs(np.sin(x1) * np.sin(x2) * fact) + 1) ** 0.1


def bartels_conn(x):
    x1, x2 = x
    return abs(x1**2 + x2**2 + x1 * x2) + abs(np.sin(x1)) + abs(np.cos(x2))


# ------------------- Benchmark Info Dictionary ------------------- #
benchmark_info = {
    "sphere": {"func": sphere, "bounds": (-100, 100), "default_dim": 50},
    "rastrigin": {"func": rastrigin, "bounds": (-5.12, 5.12), "default_dim": 30},
    "ackley": {"func": ackley, "bounds": (-32.768, 32.768), "default_dim": 30},
    "rosenbrock": {"func": rosenbrock, "bounds": (-5, 10), "default_dim": 30},
    "griewank": {"func": griewank, "bounds": (-600, 600), "default_dim": 30},
    "schwefel_226": {"func": schwefel_226, "bounds": (-500, 500), "default_dim": 30},
    "quartic_noise": {
        "func": quartic_noise,
        "bounds": (-1.28, 1.28),
        "default_dim": 30,
    },
    "powell_sum": {"func": powell_sum, "bounds": (-1, 1), "default_dim": 30},
    "schwefel_220": {"func": schwefel_220, "bounds": (-100, 100), "default_dim": 30},
    "schwefel_221": {"func": schwefel_221, "bounds": (-100, 100), "default_dim": 30},
    "step": {"func": step, "bounds": (-100, 100), "default_dim": 30},
    "stepint": {"func": stepint, "bounds": (-5.12, 5.12), "default_dim": 30},
    "schwefel_120": {"func": schwefel_120, "bounds": (-100, 100), "default_dim": 30},
    "schwefel_222": {"func": schwefel_222, "bounds": (-10, 10), "default_dim": 30},
    "schwefel_223": {"func": schwefel_223, "bounds": (-10, 10), "default_dim": 30},
    "brown": {"func": brown, "bounds": (-1, 4), "default_dim": 30},
    "dixon_price": {"func": dixon_price, "bounds": (-10, 10), "default_dim": 30},
    "powell_singular": {"func": powell_singular, "bounds": (-4, 5), "default_dim": 32},
    "zakharov": {"func": zakharov, "bounds": (-5, 10), "default_dim": 30},
    "xin_she_yang": {
        "func": xin_she_yang,
        "bounds": (-2 * np.pi, 2 * np.pi),
        "default_dim": 30,
    },
    "perm_od_beta": {"func": perm_od_beta, "bounds": (-30, 30), "default_dim": 10},
    "three_hump_camel": {"func": three_hump_camel, "bounds": (-5, 5), "default_dim": 2},
    "beale": {"func": beale, "bounds": (-4.5, 4.5), "default_dim": 2},
    "booth": {"func": booth, "bounds": (-10, 10), "default_dim": 2},
    "brent": {"func": brent, "bounds": (-10, 10), "default_dim": 2},
    "matyas": {"func": matyas, "bounds": (-10, 10), "default_dim": 2},
    "schaffer_n4": {"func": schaffer_n4, "bounds": (-100, 100), "default_dim": 2},
    "wayburn_seader3": {
        "func": wayburn_seader3,
        "bounds": (-500, 500),
        "default_dim": 2,
    },
    "leon": {"func": leon, "bounds": (-1.2, 1.2), "default_dim": 2},
    "periodic": {"func": periodic, "bounds": (-10, 10), "default_dim": 30},
    "qing": {"func": qing, "bounds": (-500, 500), "default_dim": 30},
    "alpine1": {"func": alpine1, "bounds": (-10, 10), "default_dim": 30},
    "xin_she_yang2": {
        "func": xin_she_yang2,
        "bounds": (-2 * np.pi, 2 * np.pi),
        "default_dim": 30,
    },
    "trignometric2": {"func": trignometric2, "bounds": (0, np.pi), "default_dim": 30},
    "salomon": {"func": salomon, "bounds": (-100, 100), "default_dim": 30},
    "styblinski_tang": {"func": styblinski_tang, "bounds": (-5, 5), "default_dim": 30},
    "xin_she_yang4": {"func": xin_she_yang4, "bounds": (-10, 10), "default_dim": 30},
    "xin_she_yangN2": {
        "func": xin_she_yangN2,
        "bounds": (-2 * np.pi, 2 * np.pi),
        "default_dim": 30,
    },
    "penalized": {"func": penalized, "bounds": (-50, 50), "default_dim": 30},
    "egg_crate": {"func": egg_crate, "bounds": (-5, 5), "default_dim": 2},
    "ackley_n3": {"func": ackley_n3, "bounds": (-32, 32), "default_dim": 2},
    "adjiman": {"func": adjiman, "bounds": (-1, 2), "default_dim": 2},
    "bird": {"func": bird, "bounds": (-2 * np.pi, 2 * np.pi), "default_dim": 2},
    "camel_six_hump": {"func": camel_six_hump, "bounds": (-3, 3), "default_dim": 2},
    "branin_rcos": {"func": branin_rcos, "bounds": (-5, 15), "default_dim": 2},
    "hartman3": {"func": hartman3, "bounds": (0, 1), "default_dim": 3},
    "hartman6": {"func": hartman6, "bounds": (0, 1), "default_dim": 6},
    "cross_in_tray": {"func": cross_in_tray, "bounds": (-10, 10), "default_dim": 2},
    "bartels_conn": {"func": bartels_conn, "bounds": (-500, 500), "default_dim": 2},
}

# ------------------- Experiment Runner ------------------- #


def run_pso_experiment(func, bounds, dimension, runs=20):
    results = []
    for _ in range(runs):
        _, best_val = pso_numpy(func, dimension=dimension, bounds=bounds)
        results.append(best_val)
    results = np.array(results)
    return {"mean": np.mean(results), "min": np.min(results), "std": np.std(results)}


def run_all_benchmarks(runs=20):
    results = {}

    for name, info in benchmark_info.items():
        print(f"Running {name}...")
        try:
            stats = run_pso_experiment(
                func=info["func"],
                bounds=info["bounds"],
                dimension=info["default_dim"],
                runs=runs,
            )
            results[name] = stats
        except Exception as e:
            print(f"Error running {name}: {e}")
            results[name] = {"mean": np.nan, "min": np.nan, "std": np.nan}

    return results


def create_results_table(results):
    # Sort by minimum value (best performance)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["min"])

    # Create table
    print("\n" + "=" * 80)
    print(
        f"{'Benchmark Function':<30} {'Best':<15} {'Mean':<15} {'Std':<15} {'Rank':<5}"
    )
    print("=" * 80)

    for rank, (name, stats) in enumerate(sorted_results, 1):
        if np.isnan(stats["min"]):
            print(f"{name:<30} {'N/A':<15} {'N/A':<15} {'N/A':<15} {rank:<5}")
        else:
            print(
                f"{name:<30} {stats['min']:<15.6e} {stats['mean']:<15.6e} {stats['std']:<15.6e} {rank:<5}"
            )

    print("=" * 80)


def generate_markdown_table(results, output_file="benchmark_results.md"):
    sorted_results = sorted(results.items(), key=lambda x: x[1]["min"])

    lines = []
    lines.append(
        "| Benchmark Function              | Best             | Mean             | Std              | Rank |"
    )
    lines.append(
        "|--------------------------------|------------------|------------------|------------------|------|"
    )

    for rank, (name, stats) in enumerate(sorted_results, 1):
        if np.isnan(stats["min"]):
            lines.append(
                f"| {name:<30} | N/A              | N/A              | N/A              | {rank:>4} |"
            )
        else:
            lines.append(
                f"| {name:<30} | {stats['min']:.6e} | {stats['mean']:.6e} | {stats['std']:.6e} | {rank:>4} |"
            )

    with open(output_file, "w") as f:
        f.write("# PSO Benchmark Results\n\n")
        f.write("\n".join(lines))

    print(f"Markdown results saved to {output_file}")


def convert_markdown_to_pdf(
    md_file="benchmark_results.md", pdf_file="benchmark_results.pdf"
):
    with open(md_file, "r") as f:
        md_content = f.read()

    html_content = markdown.markdown(md_content, extensions=["tables"])

    # Add CSS for better table formatting
    style = """
    <style>
        body {
            font-family: Arial, sans-serif;
            font-size: 12px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #444;
            padding: 6px 10px;
            text-align: left;
            font-size: 11px;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #fafafa;
        }
    </style>
    """

    html_full = (
        f"<!DOCTYPE html><html><head>{style}</head><body>{html_content}</body></html>"
    )

    HTML(string=html_full).write_pdf(pdf_file)
    print(f"PDF saved to {pdf_file}")


# ------------------- Main Execution ------------------- #

if __name__ == "__main__":
    # Run all benchmarks
    results = run_all_benchmarks(runs=20)

    # Create and display the results table
    create_results_table(results)

    # Generate markdown table
    generate_markdown_table(results)

    # Optional: Convert to PDF
    convert_markdown_to_pdf()

    # Save results to a file
    np.save("benchmark_results.npy", results)
    print("Results saved to benchmark_results.npy")
