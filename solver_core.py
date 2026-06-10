import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import logging
import gc
from docplex.mp.model import Model

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - OR-SOLVER: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# DFC dimensions (100x100x100 cm default normalized)
DFC_DIMS = (300, 200, 100)

class ColumnGenerationPacker:
    """
    Advanced Column Generation formulation for 3D Bin Packing 
    utilizing IBM CPLEX (docplex) for Restricted Master Problem continuous relaxation.
    """
    def __init__(self, bin_dims):
        self.bin_dims = bin_dims
        self.patterns = []
        self.sku_data = None
        self.dual_values = None
        self.stability_center = None
        self.trust_region = 0.5
        self.convergence_count = 0
        self.rmp_model = None
        self.bin_constraint = None
        self.bin_dual = 0.0

    def load_skus(self, file_path, n_skus=70):
        if not os.path.exists(file_path):
            logger.error(f"Dataset not found at target path: {file_path}")
            raise FileNotFoundError("System requires valid CSV dataset.")
            
        logger.info(f"Ingesting SKU dataset from {file_path}")
        df = pd.read_csv(file_path)
        required_cols = ['Material Barcode', 'Length', 'Width', 'Height', 'Qty']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Dataset schema violation. Missing: {[c for c in required_cols if c not in df.columns]}")
            
        sku_data = df.head(n_skus).copy()
        sku_data['Volume'] = sku_data['Length'] * sku_data['Width'] * sku_data['Height']
        sku_data['ID'] = range(len(sku_data))
        return sku_data

    def initialize_patterns(self):
        self.patterns = []
        for _, sku in self.sku_data.iterrows():
            pattern = np.zeros(len(self.sku_data))
            pattern[int(sku['ID'])] = 1
            self.patterns.append(pattern)

    def build_rmp(self):
        self.rmp_model = Model('RMP')
        self.lambda_vars = self.rmp_model.continuous_var_list(
            len(self.patterns), lb=0, name=lambda i: f'lambda_{i}'
        )
        
        pattern_sums = [np.sum(p) for p in self.patterns]
        self.rmp_model.maximize(self.rmp_model.sum(
            self.lambda_vars[i] * pattern_sums[i] for i in range(len(self.patterns)))
        )
        
        self.demand_constraints = []
        for i in range(len(self.sku_data)):
            coeffs = [p[i] for p in self.patterns]
            ct = self.rmp_model.sum(
                coeffs[j] * self.lambda_vars[j] for j in range(len(self.patterns))) <= 1
            self.demand_constraints.append(ct)
            self.rmp_model.add_constraint(ct, f'demand_{i}')
            
        self.bin_constraint = self.rmp_model.add_constraint(
            self.rmp_model.sum(self.lambda_vars) <= 1, "single_bin"
        )
        
        # Dual stabilization
        if self.stability_center is not None:
            for idx, dual_val in enumerate(self.stability_center):
                self.rmp_model.add_constraint(
                    self.demand_constraints[idx].dual_value >= dual_val - self.trust_region, f'stab_lb_{idx}'
                )
                self.rmp_model.add_constraint(
                    self.demand_constraints[idx].dual_value <= dual_val + self.trust_region, f'stab_ub_{idx}'
                )

    def solve_rmp(self):
        if not self.rmp_model:
            self.build_rmp()
        
        solution = self.rmp_model.solve()
        if not solution:
            logger.error("CPLEX failed to find continuous relaxation.")
            raise RuntimeError("RMP infeasible.")
        
        new_duals = [ct.dual_value for ct in self.demand_constraints]
        
        # Iterative stability smoothing
        if self.stability_center is None:
            self.stability_center = new_duals
        else:
            alpha = 0.7 
            self.stability_center = [alpha * new + (1 - alpha) * old for new, old in zip(new_duals, self.stability_center)]
        
        if self.convergence_count > 3:
            self.trust_region *= 0.8
            self.convergence_count = 0
            
        self.dual_values = new_duals
        self.bin_dual = self.bin_constraint.dual_value
        return solution.get_objective_value()

    def solve_psp(self, num_patterns=3):
        patterns = []
        for _ in range(num_patterns):
            if self.stability_center is not None:
                attractiveness = 1 - np.array(self.stability_center)
            else:
                attractiveness = np.ones(len(self.sku_data))
                
            sorted_skus = self.sku_data.copy()
            sorted_skus['Attractiveness'] = attractiveness
            sorted_skus = sorted_skus.sort_values('Attractiveness', ascending=False)
            
            pattern, placements = self.greedy_packing(sorted_skus)
            improved_pattern, _ = self.local_search_improvement(pattern, placements)
            patterns.append(improved_pattern)
        return patterns

    def greedy_packing(self, sorted_skus):
        """Strictly iterative multidimensional knapsack packing to prevent call-stack overflow."""
        placements = []
        occupied = np.zeros(tuple(int(x) for x in self.bin_dims), dtype=bool)
        L, W, H = (int(x) for x in self.bin_dims)
        
        def check_overlap(new_box):
            x1, y1, z1 = int(new_box['x']), int(new_box['y']), int(new_box['z'])
            x2, y2, z2 = x1 + int(new_box['Length']), y1 + int(new_box['Width']), z1 + int(new_box['Height'])
            if x2 > L or y2 > W or z2 > H: return True
            return occupied[x1:x2, y1:y2, z1:z2].any()
        
        pattern = np.zeros(len(self.sku_data))
        for _, sku in sorted_skus.iterrows():
            l, w, h = float(sku['Length']), float(sku['Width']), float(sku['Height'])
            placed = False
            for rotation in [(l,w,h), (l,h,w), (w,l,h), (w,h,l), (h,l,w), (h,w,l)]:
                dim_l, dim_w, dim_h = rotation
                if dim_l > L or dim_w > W or dim_h > H: continue
                
                for x in range(0, int(L - dim_l) + 1, 5):
                    for y in range(0, int(W - dim_w) + 1, 5):
                        for z in range(0, int(H - dim_h) + 1, 5):
                            new_box = {'x': x, 'y': y, 'z': z, 'Length': int(dim_l), 'Width': int(dim_w), 'Height': int(dim_h)}
                            if not check_overlap(new_box):
                                placements.append({'SKU': sku['Material Barcode'], 'ID': int(sku['ID']), **new_box, 'Volume': sku['Volume']})
                                occupied[x:x+int(dim_l), y:y+int(dim_w), z:z+int(dim_h)] = True
                                pattern[int(sku['ID'])] = 1
                                placed = True
                                break
                        if placed: break
                    if placed: break
                if placed: break
        
        # Explicit memory flush for continuous generation runs
        del occupied
        gc.collect()
        return pattern, placements

    def local_search_improvement(self, pattern, placements):
        unpacked_ids = [i for i in range(len(self.sku_data)) if pattern[i] == 0]
        if not unpacked_ids: return pattern, placements

        L, W, H = (int(x) for x in self.bin_dims)
        occupied = np.zeros((L, W, H), dtype=bool)
        for box in placements:
            occupied[int(box['x']):int(box['x'])+int(box['Length']), 
                     int(box['y']):int(box['y'])+int(box['Width']), 
                     int(box['z']):int(box['z'])+int(box['Height'])] = True
        
        unpacked_skus = self.sku_data.iloc[unpacked_ids].sort_values('Volume', ascending=False)
        for _, sku in unpacked_skus.iterrows():
            l, w, h = float(sku['Length']), float(sku['Width']), float(sku['Height'])
            placed = False
            for rotation in [(l,w,h), (l,h,w), (w,l,h), (w,h,l), (h,l,w), (h,w,l)]:
                dim_l, dim_w, dim_h = (int(r) for r in rotation)
                if dim_l > L or dim_w > W or dim_h > H: continue
                
                for x in [0, L - dim_l]:
                    for y in [0, W - dim_w]:
                        for z in [0, H - dim_h]:
                            if not occupied[x:x+dim_l, y:y+dim_w, z:z+dim_h].any():
                                pattern[int(sku['ID'])] = 1
                                placements.append({'SKU': sku['Material Barcode'], 'ID': int(sku['ID']), 'x': x, 'y': y, 'z': z, 'Length': dim_l, 'Width': dim_w, 'Height': dim_h, 'Volume': sku['Volume']})
                                occupied[x:x+dim_l, y:y+dim_w, z:z+dim_h] = True
                                placed = True
                                break
                        if placed: break
                    if placed: break
                if placed: break
                
        del occupied
        gc.collect()
        return pattern, placements

    def column_generation(self, max_iter=20):
        self.initialize_patterns()
        logger.info(f"Optimization pipeline initialized with {len(self.patterns)} base configurations.")
        
        best_skus, prev_obj = 0, -float('inf')
        total_start_time = time.time()
        
        for iter in range(max_iter):
            iter_start_time = time.time()
            self.build_rmp()
            num_skus = self.solve_rmp()
            logger.info(f"Iteration {iter+1}/{max_iter} | RMP Continuous Objective: {num_skus:.4f}")
            
            new_patterns = self.solve_psp(num_patterns=3)
            added = False
            
            for pattern in new_patterns:
                if self.dual_values is not None:
                    reduced_cost = np.sum(pattern) - np.dot(pattern, self.dual_values) - self.bin_dual
                    if reduced_cost < -1e-4:
                        self.patterns.append(pattern)
                        added = True
            
            for pattern in new_patterns:
                _, placements = self.greedy_packing(self.sku_data)
                if np.sum(pattern) > best_skus:
                    best_skus = np.sum(pattern)
            
            if abs(num_skus - prev_obj) < 1e-4:
                self.convergence_count += 1
                if self.convergence_count >= 3:
                    logger.info("Pricing threshold met. Optimal continuous bounds established.")
                    break
            else:
                self.convergence_count, prev_obj = 0, num_skus
                
            if not added:
                logger.info("No negative reduced cost columns identified. Subproblem space exhausted.")
                break
                
        logger.info(f"Delayed Column Generation complete. Pipeline execution time: {time.time() - total_start_time:.2f}s")
        return best_skus

if __name__ == "__main__":
    CSV_FILE = "ned_picklist_dataset_wicleath_config.csv"
    try:
        packer = ColumnGenerationPacker(DFC_DIMS)
        sku_data = packer.load_skus(CSV_FILE, 70)
        packer.sku_data = sku_data
        best_objective = packer.column_generation(max_iter=15)
    except Exception as e:
        logger.error(f"Pipeline failure: {str(e)}", exc_info=True)
