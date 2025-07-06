# Column Generation for 1D Bin Packing using DOcplex

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from docplex.mp.model import Model
import time

# ===== USER CONFIGURATION =====
BIN_CAPACITY = 100
CSV_PATH = "/Users/manav/Desktop/Column-Generation-3D-Bin-Packing/cleaned_picklist_dataset_with_config.csv"

# ===== TERMINATION CONTROL =====
USE_REDUCED_COST = True
USE_MAX_ITER = True
USE_TIME_LIMIT = False
USE_DUAL_IMPROVEMENT = True

MAX_ITER = 500
MAX_TIME = 300  # in seconds
DUAL_IMPROVEMENT_TOL = 1e-4
NO_IMPROVEMENT_LIMIT = 3

# ===== LOAD DATA =====
df = pd.read_csv(CSV_PATH)
item_lengths = df['Length'].values
num_items = len(item_lengths)

# ===== INITIAL PATTERNS (ONE ITEM PER BIN) =====
patterns = []
for i in range(num_items):
    pattern = [0] * num_items
    pattern[i] = 1
    patterns.append(pattern)

# ===== FUNCTION: BUILD AND SOLVE RMP WITH DOCPLEX =====
def solve_rmp_docplex(patterns, duals_required=True, integer=False):
    model = Model("RMP")
    x_vars = model.integer_var_list(len(patterns), lb=0, name="x") if integer else model.continuous_var_list(len(patterns), lb=0, name="x")

    # Constraints: each item must be packed at least once
    constraints = []
    for i in range(num_items):
        constraint = model.add_constraint(
            model.sum(x_vars[j] * patterns[j][i] for j in range(len(patterns))) >= 1,
            f"cover_{i}"
        )
        constraints.append(constraint)

    # Objective: Minimize number of bins
    model.minimize(model.sum(x_vars))
    solution = model.solve()

    if not solution:
        raise RuntimeError("RMP could not be solved")

    x_vals = solution.get_values(x_vars)
    duals = [c.dual_value for c in constraints] if duals_required else None

    return x_vals, duals, model

# ===== FUNCTION: SOLVE PRICING PROBLEM (KNAPSACK) =====
def solve_pricing_problem(duals):
    n = len(duals)
    dp = [0] * (BIN_CAPACITY + 1)
    keep = [[] for _ in range(BIN_CAPACITY + 1)]

    for i in range(n):
        size = int(item_lengths[i])
        value = duals[i]
        for c in range(BIN_CAPACITY, size - 1, -1):
            if dp[c - size] + value > dp[c]:
                dp[c] = dp[c - size] + value
                keep[c] = keep[c - size] + [i]

    best_value = max(dp)
    best_config = keep[dp.index(best_value)]
    return best_value, best_config

# ===== ITERATIVE COLUMN GENERATION LOOP =====
iteration = 0
start_time = time.time()
no_improve_count = 0
prev_obj_val = float("inf")

while True:
    iteration += 1
    print(f"\nIteration {iteration}: Solving RMP with {len(patterns)} patterns")
    solution, duals, model = solve_rmp_docplex(patterns, duals_required=True)
    current_obj = model.objective_value

    value, config = solve_pricing_problem(duals)
    reduced_cost = 1 - value

    # ===== Check Termination Conditions =====
    stop = False
    if USE_REDUCED_COST and reduced_cost >= -1e-6:
        print("No new column with negative reduced cost. Stopping.")
        stop = True
    if USE_MAX_ITER and iteration >= MAX_ITER:
        print(f"Reached max iteration limit ({MAX_ITER}). Stopping.")
        stop = True
    if USE_TIME_LIMIT and (time.time() - start_time) >= MAX_TIME:
        print(f"Time limit of {MAX_TIME} seconds reached. Stopping.")
        stop = True
    if USE_DUAL_IMPROVEMENT:
        if abs(prev_obj_val - current_obj) < DUAL_IMPROVEMENT_TOL:
            no_improve_count += 1
            if no_improve_count >= NO_IMPROVEMENT_LIMIT:
                print(f"No significant improvement in dual objective for {NO_IMPROVEMENT_LIMIT} iterations. Stopping.")
                stop = True
        else:
            no_improve_count = 0
        prev_obj_val = current_obj

    if stop:
        break

    print(f"Adding new pattern with reduced cost {reduced_cost:.4f}")
    new_pattern = [0] * num_items
    for i in config:
        new_pattern[i] = 1
    patterns.append(new_pattern)

# ===== FINAL INTEGER SOLVE =====
final_solution, _, _ = solve_rmp_docplex(patterns, duals_required=False, integer=True)

# ===== METRICS AND VISUALIZATION =====
used_bins = []
for j, val in enumerate(final_solution):
    if int(round(val)) > 0:
        for _ in range(int(round(val))):
            used_bins.append(patterns[j])

total_bins = len(used_bins)
total_length = sum(item_lengths)
fill_levels = []

print("\n===== FINAL METRICS =====")
print(f"Total bins used: {total_bins}")

for b in used_bins:
    fill = sum(item_lengths[i] for i, used in enumerate(b) if used)
    fill_levels.append(fill)

avg_fill = np.mean(fill_levels)
print(f"Average fill per bin: {avg_fill:.2f}")
print(f"Fill rate: {avg_fill / BIN_CAPACITY * 100:.2f}%")

plt.figure(figsize=(10, 6))
plt.hist(fill_levels, bins=10, color='lightgreen', edgecolor='black')
plt.title("Histogram of Bin Fill Levels")
plt.xlabel("Fill Level")
plt.ylabel("Number of Bins")
plt.grid(True)
plt.tight_layout()
plt.savefig("final_bin_fill_distribution.png")  # Save the final plot
plt.show()

# ===== SAVE FINAL RESULTS TO CSV =====
pd.DataFrame(final_solution, columns=["Bin_Count"]).to_csv("final_bin_solution.csv", index_label="Pattern_Index")
print("\nSaved final solution to 'final_bin_solution.csv'")
