﻿import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import funcs_vec
from IPython.display import display
from matplotlib.backends.backend_pdf import PdfPages

# GA parameters
CROSSOVER_PROB = 0.75
MUTATION_PROB = 0.02
POP_SIZE = 50
MAX_EVALS = 40000
TOURNAMENT_SIZE = 3
N_RUNS = 20

rng_global = np.random.default_rng()


# ---------------- GA FUNCTION ----------------
def run_ga_vectorized(func, lower, upper, seed=None):
    rng = np.random.default_rng(seed)
    dim = int(lower.shape[0])
    pop = rng.uniform(lower, upper, size=(POP_SIZE, dim))
    fitness = func(pop)
    evals = POP_SIZE
    best_idx = int(np.argmin(fitness))
    best_val = float(fitness[best_idx])
    best_x = pop[best_idx].copy()

    while evals < MAX_EVALS:
        # tournament selection
        cand = rng.integers(0, POP_SIZE, size=(POP_SIZE, TOURNAMENT_SIZE))
        cand_f = fitness[cand]
        winners = cand[np.arange(POP_SIZE), np.argmin(cand_f, axis=1)]
        p1 = pop[winners]

        cand2 = rng.integers(0, POP_SIZE, size=(POP_SIZE, TOURNAMENT_SIZE))
        cand_f2 = fitness[cand2]
        winners2 = cand2[np.arange(POP_SIZE), np.argmin(cand_f2, axis=1)]
        p2 = pop[winners2]

        # crossover
        do_x = rng.random(size=POP_SIZE) < CROSSOVER_PROB
        alpha = rng.random(size=(POP_SIZE, dim))
        children = np.where(do_x[:, None], alpha * p1 + (1 - alpha) * p2, p1.copy())

        # mutation
        mut_mask = rng.random(size=(POP_SIZE, dim)) < MUTATION_PROB
        if mut_mask.any():
            sigma = 0.1 * (upper - lower)
            noise = rng.normal(0, 1, size=(POP_SIZE, dim)) * sigma
            children = np.where(mut_mask, children + noise, children)

        children = np.clip(children, lower, upper)
        child_f = func(children)
        evals += POP_SIZE

        # update best
        idx = int(np.argmin(child_f))
        if child_f[idx] < best_val:
            best_val = float(child_f[idx])
            best_x = children[idx].copy()

        pop = children
        fitness = child_f

    return best_val, best_x


# ---------------- PDF FUNCTION ----------------
def save_df_to_pdf_matplotlib(df, pdf_path, title="GA Benchmark Results"):
    fig, ax = plt.subplots(
        figsize=(max(8, 0.6 * len(df.columns)), max(4, 0.5 * len(df)))
    )
    ax.axis("tight")
    ax.axis("off")

    # Convert list columns to strings for display
    df_display = df.copy()
    for col in df_display.columns:
        df_display[col] = df_display[col].apply(
            lambda x: str(x) if isinstance(x, list) else x
        )

    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    plt.title(title, fontsize=14)

    pp = PdfPages(pdf_path)
    pp.savefig(fig, bbox_inches="tight")
    pp.close()
    plt.close()
    print("Saved PDF to:", pdf_path)


# ---------------- MAIN ----------------
summary_rows = []
all_results = []
print("Starting vectorized GA runs...")

for name, func, low, high, known in funcs_vec:
    best_vals = np.empty(N_RUNS)
    best_xs = []

    for i in range(N_RUNS):
        seed = rng_global.integers(1_000_000_000)
        try:
            bv, bx = run_ga_vectorized(func, low, high, seed=seed)
        except Exception as e:
            print(f"  Skipping run due to error evaluating {name}: {e}")
            bv, bx = np.nan, None
        best_vals[i] = bv
        best_xs.append(bx)

    valid_vals = best_vals[~np.isnan(best_vals)]
    if len(valid_vals) > 0:
        meanv = float(np.nanmean(best_vals))
        stdv = float(np.nanstd(best_vals, ddof=1)) if len(valid_vals) > 1 else 0.0
        idx_best = int(np.nanargmin(best_vals))
        best_observed_val = float(best_vals[idx_best])
        best_observed_x = (
            np.round(best_xs[idx_best], 6).tolist()
            if best_xs[idx_best] is not None
            else None
        )
    else:
        meanv = np.nan
        stdv = np.nan
        best_observed_val = np.nan
        best_observed_x = None

    summary_rows.append(
        {
            "function": name,
            "mean_best": round(meanv, 6) if not np.isnan(meanv) else np.nan,
            "std_best": round(stdv, 6) if not np.isnan(stdv) else np.nan,
            "best_observed_val": best_observed_val,
            "best_observed_x": best_observed_x,
            "known_min": known,
        }
    )

    all_results.append({"function": name, "best_vals": best_vals, "best_xs": best_xs})

    if len(valid_vals) > 0:
        print(
            f"{name}: mean={meanv:.6g}, std={stdv:.6g}, best={np.nanmin(best_vals):.6g}"
        )
    else:
        print(f"{name}: All runs failed - no valid results")

df_summary = pd.DataFrame(summary_rows)
display(df_summary)

# Per-run table
rows = []
for r in all_results:
    row = {"function": r["function"]}
    for i, val in enumerate(r["best_vals"], start=1):
        row[f"run_{i}"] = float(np.round(val, 8)) if not np.isnan(val) else np.nan
    rows.append(row)

df_runs = pd.DataFrame(rows)
display(df_runs)

# ---------------- SAVE PDFs ----------------
save_df_to_pdf_matplotlib(
    df_summary,
    "ga_benchmark_results_summary_vectorized.pdf",
    title="GA Benchmark Summary",
)
save_df_to_pdf_matplotlib(
    df_runs,
    "ga_benchmark_results_runs_vectorized.pdf",
    title="GA Benchmark Per-Run Results",
)
