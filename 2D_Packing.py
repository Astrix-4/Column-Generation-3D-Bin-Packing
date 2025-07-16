# ===============================================
# 2D Bin Packing via Column Generation with EA & CPLEX
# Based on Puchinger & Raidl (2004) GECCO Paper
# ===============================================

# SECTION 1: Imports & Configuration
import os
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
from docplex.mp.model import Model

# SECTION 2: Global Parameters (easily configurable)
BIN_WIDTH = 100
BIN_HEIGHT = 100
MAX_ITERATIONS = 800
MAX_TIME = 600  # in seconds
MAX_ITEMS = None  # Set to an integer to limit the number of items loaded
ENABLE_EA = True  # Set to False to disable EA
ENABLE_ILP_FALLBACK = True  # Set to False to disable ILP fallback
RANDOM_SEED = 42
EA_POP_SIZE = 100
EA_MUTATION_PROB = 0.75
EA_MAX_GENERATIONS = 1000
EA_MAX_NO_IMPROVE = 200

INPUT_CSV = "/Users/manav/Desktop/Column-Generation-3D-Bin-Packing/cleaned_picklist_dataset_with_config.csv"
OUTPUT_FOLDER = "2D_Visualization"
REPORT_FILE = "2D_Visualization/final_report.csv"

ALLOW_ROTATION = True  # Set to False to disable rotation
random.seed(RANDOM_SEED)
import shutil
if os.path.exists(OUTPUT_FOLDER):
    for filename in os.listdir(OUTPUT_FOLDER):
        if filename.endswith(".jpg"):
            os.remove(os.path.join(OUTPUT_FOLDER, filename))
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# SECTION 3: Data Structures
class Item:
    def __init__(self, id, width, height):
        self.id = id
        self.width = width
        self.height = height

    def get_orientations(self):
        return [(self.width, self.height), (self.height, self.width)] if ALLOW_ROTATION else [(self.width, self.height)]

class Pattern:
    def __init__(self, items):
        self.items = items  # list of tuples (Item, width, height)
        self.width = max([w for (_, w, _) in items]) if items else 0
        self.height = sum([h for (_, _, h) in items])
        self.area = self.width * self.height

    def total_profit(self, duals):
        return sum([duals.get(i.id, 0) for (i, _, _) in self.items])

    def __repr__(self):
        return f"Pattern(items={[i.id for i, _, _ in self.items]}, height={self.height}, width={self.width})"

# SECTION 4: Load Items from CSV
def load_items(max_items=None):
    df = pd.read_csv(INPUT_CSV)
    if max_items is not None:
        df = df.head(max_items)
    items = []
    for idx, row in df.iterrows():
        w = int(row['Width'])
        h = int(row['Length'])  # Length used as height
        items.append(Item(idx, w, h))
    return items

# SECTION 5: Visualization and Report
def visualize_bins(bin_patterns):
    report_data = []
    for i, pattern in enumerate(bin_patterns):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, BIN_WIDTH)
        ax.set_ylim(0, BIN_HEIGHT)
        total_item_area = sum(w * h for _, w, h in pattern.items)
        bin_area = BIN_WIDTH * BIN_HEIGHT
        efficiency = total_item_area / bin_area * 100
        ax.set_title(f"Bin {i+1} - Efficiency: {efficiency:.2f}%")
        ax.set_aspect('equal')
        ax.axis('off')

        y_offset = 0
        for item, w, h in pattern.items:
            rect = Rectangle((0, y_offset), w, h, linewidth=1, edgecolor='black', facecolor='lightblue')
            ax.add_patch(rect)
            ax.text(w / 2, y_offset + h / 2, f"ID {item.id}{' (R)' if w != item.width else ''}", ha='center', va='center', fontsize=8)
            report_data.append({"Bin": i+1, "ItemID": item.id, "Width": w, "Height": h, "Y_Offset": y_offset, "Efficiency(%)": round(efficiency, 2)})
            y_offset += h

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"bin_{i+1:03d}.jpg"))
        plt.close()

    df_report = pd.DataFrame(report_data)
    df_report.to_csv(REPORT_FILE, index=False)

# SECTION 6: Greedy FFBC Heuristic
def greedy_generate(items, duals):
    patterns = []
    unplaced = set(items)
    while unplaced:
        pattern_items = []
        h_used = 0
        for item in sorted(unplaced, key=lambda x: -duals.get(x.id, 0)):
            placed = False
            for w, h in item.get_orientations():
                if h_used + h <= BIN_HEIGHT and w <= BIN_WIDTH:
                    pattern_items.append((item, w, h))
                    h_used += h
                    placed = True
                    break
            if placed:
                unplaced.remove(item)
        if not pattern_items:
            break
        patterns.append(Pattern(pattern_items))
    return patterns

# SECTION 7: EA-based Pricing Heuristic
def ea_generate(items, duals):
    def fitness(pattern):
        return pattern.total_profit(duals)

    def mutate(pattern):
        new_items = [i for i, _, _ in pattern.items]
        if new_items:
            idx = random.randint(0, len(new_items) - 1)
            del new_items[idx]
        random.shuffle(new_items)
        return build_pattern(new_items)

    def build_pattern(candidate_items):
        packed = []
        height_used = 0
        for item in candidate_items:
            for w, h in item.get_orientations():
                if height_used + h <= BIN_HEIGHT and w <= BIN_WIDTH:
                    packed.append((item, w, h))
                    height_used += h
                    break
        return Pattern(packed)

    population = [build_pattern(random.sample(items, len(items))) for _ in range(EA_POP_SIZE)]
    best = max(population, key=fitness)
    no_improve = 0
    for gen in range(EA_MAX_GENERATIONS):
        if no_improve >= EA_MAX_NO_IMPROVE:
            break
        child = mutate(best)
        if fitness(child) > fitness(best):
            best = child
            no_improve = 0
        else:
            no_improve += 1
    if best.total_profit(duals) > 1.0001:
        return best
    return None

# SECTION 8: ILP Pricing Problem
def solve_ilp_pricing(items, duals):
    mdl = Model(name="PricingProblemWithRotation")
    x = {i.id: mdl.binary_var(name=f"x_{i.id}") for i in items}
    xr = {i.id: mdl.binary_var(name=f"xr_{i.id}") for i in items}

    mdl.add_constraint(mdl.sum(i.width * x[i.id] + i.height * xr[i.id] for i in items) <= BIN_WIDTH)
    mdl.add_constraint(mdl.sum(i.height * x[i.id] + i.width * xr[i.id] for i in items) <= BIN_HEIGHT)
    for i in items:
        mdl.add_constraint(x[i.id] + xr[i.id] <= 1)

    mdl.maximize(mdl.sum(duals[i.id] * (x[i.id] + xr[i.id]) for i in items))
    sol = mdl.solve(log_output=False)

    if sol and sol.get_objective_value() > 1 + 1e-5:
        selected = []
        for i in items:
            if sol.get_value(x[i.id]) > 0.5:
                selected.append((i, i.width, i.height))
            elif sol.get_value(xr[i.id]) > 0.5:
                selected.append((i, i.height, i.width))
        return Pattern(selected)
    return None

# SECTION 9: Column Generation Loop
USE_ONLY_ILP = False  # Set to True to force ILP-only pattern generation

def column_generation(items):
    patterns = greedy_generate(items, defaultdict(lambda: 1))
    iteration = 0
    start_time = time.time()
    while iteration < MAX_ITERATIONS and time.time() - start_time < MAX_TIME:
        mdl = Model(name=f"Master_{iteration}")
        x = [mdl.continuous_var(lb=0, ub=1, name=f"x_{i}") for i in range(len(patterns))]
        for item in items:
            mdl.add_constraint(mdl.sum(x[i] for i in range(len(patterns)) if item.id in [itm.id for itm, _, _ in patterns[i].items]) >= 1)
        mdl.minimize(mdl.sum(x))
        sol = mdl.solve()
        if not sol:
            print("[Warning] Master problem couldn't be solved. Breaking.")
            break
        duals = {item.id: c.dual_value for item, c in zip(items, mdl.iter_constraints())}

        new_pattern = None
        method_used = None

        if USE_ONLY_ILP:
            new_pattern = solve_ilp_pricing(items, duals)
            if new_pattern:
                method_used = "ILP-ONLY"
        else:
            if ENABLE_EA:
                new_pattern = ea_generate(items, duals)
                if new_pattern:
                    method_used = "EA"

            if not new_pattern and ENABLE_ILP_FALLBACK:
                new_pattern = solve_ilp_pricing(items, duals)
                if new_pattern:
                    method_used = "ILP"

        if not new_pattern:
            print(f"[Info] No improving column found at iteration {iteration}. Terminating.")
            break
        else:
            profit = new_pattern.total_profit(duals)
            rc = 1 - profit
            if profit <= 1 + 1e-5:
                print(f"[Stop] No improving column (profit = {profit:.5f}).")
                break
            print(f"[Log] Dual profits: {[round(duals[item.id], 3) for item in items[:5]]}...")
            print(f"[Log] Pattern Profit: {profit:.5f}")
            print(f"[Info] Iteration {iteration}: Added pattern (reduced cost = {rc:.4f}) via {method_used}")
            patterns.append(new_pattern)
        iteration += 1

    total_time = time.time() - start_time
    print(f"[Done] Column Generation completed in {iteration} iterations and {total_time:.2f} seconds.")

    # Final ILP Solve for integrality
    print("[Final ILP] Solving master problem as ILP for integral bin selection...")
    mdl = Model(name="FinalMasterILP")
    x = [mdl.binary_var(name=f"x_{i}") for i in range(len(patterns))]
    for item in items:
        mdl.add_constraint(mdl.sum(x[i] for i in range(len(patterns)) if item.id in [itm.id for itm, _, _ in patterns[i].items]) >= 1)
    mdl.minimize(mdl.sum(x))
    solution = mdl.solve()

    selected_patterns = [patterns[i] for i in range(len(patterns)) if solution.get_value(x[i]) > 0.5]
    print(f"[Final ILP] Selected {len(selected_patterns)} bins (integral).")
    return selected_patterns

# SECTION 10: Main Entry Point
if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"CSV file '{INPUT_CSV}' not found.")
    print("[Start] Reading input items...")
    MAX_ITEMS = None  # Set to an integer to limit the number of items loaded
    items = load_items(MAX_ITEMS)
    print(f"[Info] Loaded {len(items)} items. Starting Column Generation...")
    final_patterns = column_generation(items)
    visualize_bins(final_patterns)
    print(f"[Result] Used {len(final_patterns)} bin patterns. Report saved to '{REPORT_FILE}'")
