import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
from docplex.mp.model import Model

# DFC dimensions (200x200x200 cm)
DFC_DIMS = (100, 100, 100)

class ColumnGenerationPacker:
    def __init__(self, bin_dims):
        self.bin_dims = bin_dims
        self.patterns = []
        self.sku_data = None
        self.dual_values = None
        self.rmp_model = None
        self.bin_constraint = None
        self.bin_dual = 0.0

    def load_skus(self, file_path, n_skus=50):
        """Load SKU data with robust path resolution"""
        paths_to_try = [
            file_path,
            os.path.join(os.getcwd(), file_path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path),
            os.path.join(os.path.expanduser("~/Desktop/Column-Generation-3D-Bin-Packing"), file_path)
        ]
        for path in paths_to_try:
            if os.path.exists(path):
                print(f"Found CSV file at: {path}")
                df = pd.read_csv(path)
                required_cols = ['Material Barcode', 'Length', 'Width', 'Height', 'Qty']
                if not all(col in df.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in df.columns]
                    raise ValueError(f"CSV missing required columns: {missing}")
                sku_data = df.head(n_skus).copy()
                sku_data['Volume'] = sku_data['Length'] * sku_data['Width'] * sku_data['Height']
                sku_data['ID'] = range(len(sku_data))
                return sku_data
        print("Error: CSV file not found in these locations:")
        for path in paths_to_try:
            print(f"  - {path}")
        raise FileNotFoundError("CSV file not found")

    def initialize_patterns(self):
        self.patterns = []
        for _, sku in self.sku_data.iterrows():
            pattern = np.zeros(len(self.sku_data))
            pattern[int(sku['ID'])] = 1
            self.patterns.append(pattern)

    def build_rmp(self):
        self.rmp_model = Model('RMP')
        self.lambda_vars = self.rmp_model.continuous_var_list(
            len(self.patterns),
            lb=0,
            name=lambda i: f'lambda_{i}'
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
        # Single bin constraint
        self.bin_constraint = self.rmp_model.add_constraint(
            self.rmp_model.sum(self.lambda_vars) <= 1,
            "single_bin"
        )

    def solve_rmp(self):
        if not self.rmp_model:
            self.build_rmp()
        solution = self.rmp_model.solve()
        if not solution:
            raise RuntimeError("RMP failed to solve")
        self.dual_values = [ct.dual_value for ct in self.demand_constraints]
        self.bin_dual = self.bin_constraint.dual_value
        return solution.get_objective_value()

    def solve_psp(self):
        """Pricing Subproblem with corrected attractiveness"""
        if self.dual_values is None or len(self.dual_values) != len(self.sku_data):
            print("Using fallback attractiveness (all 1s)")
            attractiveness = np.ones(len(self.sku_data))
        else:
            attractiveness = 1 - np.array(self.dual_values) - self.bin_dual
        
        sorted_skus = self.sku_data.copy()
        sorted_skus['Attractiveness'] = attractiveness
        sorted_skus = sorted_skus.sort_values('Attractiveness', ascending=False)
        placements = []
        occupied = np.zeros(tuple(int(x) for x in self.bin_dims), dtype=bool)
        L, W, H = (int(x) for x in self.bin_dims)
        
        def check_overlap(new_box):
            x1, y1, z1 = int(new_box['x']), int(new_box['y']), int(new_box['z'])
            x2 = x1 + int(new_box['Length'])
            y2 = y1 + int(new_box['Width'])
            z2 = z1 + int(new_box['Height'])
            if x2 > L or y2 > W or z2 > H:
                return True
            return occupied[x1:x2, y1:y2, z1:z2].any()
        
        pattern = np.zeros(len(self.sku_data))
        for _, sku in sorted_skus.iterrows():
            l, w, h = float(sku['Length']), float(sku['Width']), float(sku['Height'])
            placed = False
            for rotation in [(l, w, h), (l, h, w), (w, l, h),
                            (w, h, l), (h, l, w), (h, w, l)]:
                dim_l, dim_w, dim_h = (float(rotation[0]), float(rotation[1]), float(rotation[2]))
                if dim_l > L or dim_w > W or dim_h > H:
                    continue
                x_end = int(L - dim_l) + 1
                y_end = int(W - dim_w) + 1
                z_end = int(H - dim_h) + 1
                for x in range(0, x_end, 5):
                    for y in range(0, y_end, 5):
                        for z in range(0, z_end, 5):
                            new_box = {
                                'x': x, 'y': y, 'z': z,
                                'Length': int(dim_l), 'Width': int(dim_w), 'Height': int(dim_h)
                            }
                            if not check_overlap(new_box):
                                placements.append({
                                    'SKU': sku['Material Barcode'],
                                    'ID': sku['ID'],
                                    **new_box,
                                    'Volume': sku['Volume']
                                })
                                occupied[
                                    x:x+int(dim_l),
                                    y:y+int(dim_w),
                                    z:z+int(dim_h)
                                ] = True
                                pattern[int(sku['ID'])] = 1
                                placed = True
                                break
                        if placed: break
                    if placed: break
                if placed: break
        return pattern, placements

    def column_generation(self, max_iter=20):
        self.initialize_patterns()
        print(f"Initialized with {len(self.patterns)} patterns")
        print("Starting column generation iterations...")
        best_skus = 0
        best_placements = []
        for iter in range(max_iter):
            print(f"\n--- Iteration {iter+1}/{max_iter} ---")
            print("Solving RMP...")
            self.build_rmp()
            num_skus = self.solve_rmp()
            print(f"  RMP packed {num_skus:.2f} SKUs")
            print("Solving PSP...")
            new_pattern, placements = self.solve_psp()
            new_skus = np.sum(new_pattern)
            if self.dual_values is not None:
                reduced_cost = new_skus - np.dot(new_pattern, self.dual_values) - self.bin_dual
                print(f"  PSP found pattern with {new_skus} SKUs, reduced cost: {reduced_cost:.4f}")
            else:
                reduced_cost = float('-inf')
                print(f"  PSP found pattern with {new_skus} SKUs (no dual values)")
            if new_skus > best_skus:
                best_skus = new_skus
                best_placements = placements
                print(f"  New best solution: {best_skus} SKUs")
            if reduced_cost <= 1e-4:
                print("Termination condition met (reduced cost <= 1e-4).")
                break
            self.patterns.append(new_pattern)
            print(f"  Added new pattern")
        print("\nColumn generation completed successfully!")
        return best_placements, best_skus

def visualize_packing(placements, bin_dims):
    """Create 3D visualization of packing result"""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    L, W, H = bin_dims
    
    # Configure 3D axes
    ax.set_box_aspect([L, W, H])
    ax.set_xlabel('Length (cm)', fontsize=12, labelpad=15)
    ax.set_ylabel('Width (cm)', fontsize=12, labelpad=15)
    ax.set_zlabel('Height (cm)', fontsize=12, labelpad=15)
    ax.set_xlim([0, L])
    ax.set_ylim([0, W])
    ax.set_zlim([0, H])
    
    # Draw container wireframe
    for edge in [
        [(0,0,0), (L,0,0)], [(0,0,0), (0,W,0)], [(0,0,0), (0,0,H)],
        [(L,W,H), (0,W,H)], [(L,W,H), (L,0,H)], [(L,W,H), (L,W,0)],
        [(L,0,0), (L,W,0)], [(L,0,0), (L,0,H)],
        [(0,W,0), (L,W,0)], [(0,W,0), (0,W,H)],
        [(0,0,H), (L,0,H)], [(0,0,H), (0,W,H)]
    ]:
        ax.plot3D(*zip(*edge), 'k--', alpha=0.3, linewidth=1)
    
    # Draw packed SKUs
    if placements:
        colors = plt.cm.tab20(np.linspace(0, 1, len(placements)))
        for i, box in enumerate(placements):
            ax.bar3d(box['x'], box['y'], box['z'],
                     box['Length'], box['Width'], box['Height'],
                     alpha=0.7, color=colors[i], edgecolor='black')
            
            # Add SKU label
            label = str(box['SKU'])[-4:]
            ax.text(box['x'] + box['Length']/2,
                    box['y'] + box['Width']/2,
                    box['z'] + box['Height']/2,
                    label, color='black', fontsize=8, ha='center')
    
    ax.set_title(f'Column Generation Packing: {len(placements) if placements else 0} SKUs in DFC',
                 fontsize=16, pad=20)
    plt.tight_layout()
    return fig

def calculate_metrics(placements, bin_dims, total_skus):
    """Calculate comprehensive packing metrics"""
    packed_volume = sum(p['Volume'] for p in placements) if placements else 0
    bin_volume = bin_dims[0] * bin_dims[1] * bin_dims[2]
    volume_utilization = packed_volume / bin_volume
    sku_packing_rate = len(placements) / total_skus if total_skus > 0 else 0
    
    print("=" * 65)
    print("COLUMN GENERATION PACKING RESULTS")
    print("=" * 65)
    print(f"Total SKUs available: {total_skus}")
    print(f"SKUs successfully packed: {len(placements)}")
    print(f"SKU packing efficiency: {sku_packing_rate:.1%}")
    print(f"Volume utilization: {volume_utilization:.2%}")
    print("-" * 65)
    print(f"Packed volume: {packed_volume:,.0f} cm³")
    print(f"Available volume: {bin_volume:,.0f} cm³")
    print(f"Wasted space: {bin_volume - packed_volume:,.0f} cm³")
    print("=" * 65)
    
    if placements:
        # Space efficiency metrics
        extents = {
            'x_max': max(p['x'] + p['Length'] for p in placements),
            'y_max': max(p['y'] + p['Width'] for p in placements),
            'z_max': max(p['z'] + p['Height'] for p in placements)
        }
        used_volume = extents['x_max'] * extents['y_max'] * extents['z_max']
        print(f"Space utilization: {used_volume / bin_volume:.1%}")
        print(f"Effective packing density: {packed_volume / used_volume:.1%}")
        print("=" * 65)

if __name__ == "__main__":
    CSV_FILE = "cleaned_picklist_dataset_with_config.csv"
    NUM_SKUS = 50  # Changed to 50 per your last run
    
    try:
        print("=" * 70)
        print("Initializing Column Generation Packing System...")
        print("=" * 70)
        packer = ColumnGenerationPacker(DFC_DIMS)
        
        print("Loading and preprocessing SKU data...")
        sku_data = packer.load_skus(CSV_FILE, NUM_SKUS)
        packer.sku_data = sku_data
        total_skus = len(sku_data)
        print(f"Loaded {total_skus} SKUs for optimization")
        
        print("\n" + "=" * 70)
        print("Starting Column Generation Process...")
        print("=" * 70)
        
        start_time = time.time()
        placements, num_packed = packer.column_generation(max_iter=15)
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("FINAL PACKING RESULTS")
        print("=" * 70)
        print(f"  Packed SKUs: {num_packed}/{total_skus}")
        print(f"  Computation time: {elapsed_time:.2f} seconds")
        print("=" * 70)
        
        print("\nGenerating comprehensive metrics...")
        calculate_metrics(placements, DFC_DIMS, total_skus)
        
        print("\nCreating visualization...")
        fig = visualize_packing(placements, DFC_DIMS)
        
        # Save results
        output_img = "column_generation_packing_result.png"
        plt.savefig(output_img, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as: {output_img}")
        
        # Show interactive plot
        plt.show()
        
        print("\n" + "=" * 70)
        print("PROGRAM EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n" + "!" * 70)
        print(f"ERROR: {str(e)}")
        print("!" * 70)
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Ensure the CSV file exists and has correct columns")
        print("2. Check CPLEX installation: python -c 'import cplex; print(cplex.__version__)'")
        print("3. Verify your CSV has at least 50 SKUs")
