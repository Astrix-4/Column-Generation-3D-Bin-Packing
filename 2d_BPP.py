# ===============================================
# Complete Integrated 2D Bin Packing Solution
# Combines optimized algorithm with comprehensive visualization
# ===============================================

import os
import sys
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
import seaborn as sns
from docplex.mp.model import Model

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ===============================================
# SECTION 1: GLOBAL CONFIGURATION
# ===============================================

# Bin and algorithm parameters
BIN_WIDTH = 100
BIN_HEIGHT = 100
MAX_ITERATIONS = 500
MAX_TIME = 600
MAX_ITEMS = None
ENABLE_EA = True
ENABLE_ILP_FALLBACK = True
RANDOM_SEED = 42
EA_POP_SIZE = 100
EA_MUTATION_PROB = 0.75
EA_MAX_GENERATIONS = 1000
EA_MAX_NO_IMPROVE = 200

# File paths - UPDATE THESE FOR YOUR SYSTEM
INPUT_CSV = r"C:\Users\ASTEEK NARAYAN\Desktop\My_sura\cleaned_picklist_dataset_with_config (1).csv"
OUTPUT_FOLDER = "2D_Visualization"
REPORT_FILE = "2D_Visualization/final_report.csv"
CONVERGENCE_PLOT = "2D_Visualization/convergence_analysis.png"

ALLOW_ROTATION = True
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Initialize output folder
if os.path.exists(OUTPUT_FOLDER):
    for filename in os.listdir(OUTPUT_FOLDER):
        if filename.endswith((".jpg", ".png", ".gif")):
            try:
                os.remove(os.path.join(OUTPUT_FOLDER, filename))
            except:
                pass
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===============================================
# SECTION 2: DATA STRUCTURES
# ===============================================

@dataclass(frozen=True)
class Item:
    """Immutable item with efficient hashing and comparison"""
    id: int
    width: int
    height: int
    
    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Item {self.id}: dimensions must be positive")
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def get_orientations(self) -> List[Tuple[int, int]]:
        """Get all valid orientations for this item"""
        orientations = [(self.width, self.height)]
        if ALLOW_ROTATION and self.width != self.height:
            orientations.append((self.height, self.width))
        return orientations
    
    def fits_in_bin(self, available_width: int, available_height: int) -> List[Tuple[int, int]]:
        """Get orientations that fit in given space"""
        return [(w, h) for w, h in self.get_orientations() 
                if w <= available_width and h <= available_height]

@dataclass
class PackedItem:
    """Represents an item packed in a specific orientation"""
    item: Item
    width: int
    height: int
    
    def __post_init__(self):
        valid_orientations = self.item.get_orientations()
        if (self.width, self.height) not in valid_orientations:
            raise ValueError(f"Invalid orientation for item {self.item.id}")
    
    @property
    def is_rotated(self) -> bool:
        return self.width != self.item.width

@dataclass
class Pattern:
    """Optimized pattern representation with caching"""
    items: List[PackedItem] = field(default_factory=list)
    _cached_properties: Dict = field(default_factory=dict, init=False, repr=False)
    
    def __post_init__(self):
        self._validate_pattern()
    
    def _validate_pattern(self):
        """Validate that pattern fits in bin"""
        if not self.items:
            return
        
        max_width = max(item.width for item in self.items)
        total_height = sum(item.height for item in self.items)
        
        if max_width > BIN_WIDTH or total_height > BIN_HEIGHT:
            raise ValueError("Pattern exceeds bin dimensions")
    
    @property
    def width(self) -> int:
        if 'width' not in self._cached_properties:
            self._cached_properties['width'] = max([item.width for item in self.items], default=0)
        return self._cached_properties['width']
    
    @property
    def height(self) -> int:
        if 'height' not in self._cached_properties:
            self._cached_properties['height'] = sum(item.height for item in self.items)
        return self._cached_properties['height']
    
    @property
    def area(self) -> int:
        if 'area' not in self._cached_properties:
            self._cached_properties['area'] = sum(item.width * item.height for item in self.items)
        return self._cached_properties['area']
    
    @property
    def item_ids(self) -> Set[int]:
        if 'item_ids' not in self._cached_properties:
            self._cached_properties['item_ids'] = {item.item.id for item in self.items}
        return self._cached_properties['item_ids']
    
    @property
    def efficiency(self) -> float:
        """Pattern efficiency as percentage of bin area used"""
        bin_area = BIN_WIDTH * BIN_HEIGHT
        return (self.area / bin_area) * 100 if bin_area > 0 else 0
    
    def total_profit(self, duals: Dict[int, float]) -> float:
        """Calculate total profit based on dual values"""
        return sum(duals.get(item.item.id, 0) for item in self.items)
    
    def add_item(self, packed_item: PackedItem) -> bool:
        """Try to add an item to the pattern"""
        new_height = self.height + packed_item.height
        new_width = max(self.width, packed_item.width)
        
        if new_height <= BIN_HEIGHT and new_width <= BIN_WIDTH:
            self.items.append(packed_item)
            self._cached_properties.clear()  # Clear cache
            return True
        return False

@dataclass
class ColumnGenerationTracker:
    """Track column generation progress for visualization"""
    iteration: int = 0
    objective_values: List[float] = field(default_factory=list)
    reduced_costs: List[float] = field(default_factory=list)
    pattern_counts: List[int] = field(default_factory=list)
    dual_values_history: List[Dict[int, float]] = field(default_factory=list)
    method_used: List[str] = field(default_factory=list)
    computation_times: List[float] = field(default_factory=list)
    
    def add_iteration(self, obj_val: float, reduced_cost: float, num_patterns: int, 
                     duals: Dict[int, float], method: str, comp_time: float):
        """Add data for current iteration"""
        self.iteration += 1
        self.objective_values.append(obj_val)
        self.reduced_costs.append(reduced_cost)
        self.pattern_counts.append(num_patterns)
        self.dual_values_history.append(duals.copy())
        self.method_used.append(method)
        self.computation_times.append(comp_time)

# ===============================================
# SECTION 3: DATA LOADING
# ===============================================

def load_items(max_items: Optional[int] = None) -> List[Item]:
    """Load and validate items from CSV with error handling"""
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"[Info] Successfully loaded CSV with {len(df)} rows")
        
        if max_items is not None:
            df = df.head(max_items)
            print(f"[Info] Limited to first {max_items} items")
        
        # Validate required columns
        required_cols = ['Width', 'Length']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        items = []
        invalid_items = 0
        
        for idx, row in df.iterrows():
            try:
                w = int(row['Width'])
                h = int(row['Length'])  # Length used as height
                
                # Skip invalid items
                if w <= 0 or h <= 0 or w > BIN_WIDTH or h > BIN_HEIGHT:
                    invalid_items += 1
                    continue
                
                items.append(Item(idx, w, h))
            except (ValueError, TypeError) as e:
                invalid_items += 1
                continue
        
        print(f"[Info] Loaded {len(items)} valid items, skipped {invalid_items} invalid items")
        
        # Sort items by area descending for better initial patterns
        items.sort(key=lambda x: x.area, reverse=True)
        
        return items
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{INPUT_CSV}' not found.")
    except Exception as e:
        raise Exception(f"Error loading items: {str(e)}")

# ===============================================
# SECTION 4: PATTERN GENERATION
# ===============================================

class PatternGenerator:
    """Centralized pattern generation with multiple strategies"""
    
    def __init__(self, items: List[Item]):
        self.items = items
        self.item_lookup = {item.id: item for item in items}
    
    def greedy_generate(self, duals: Dict[int, float]) -> List[Pattern]:
        """Enhanced greedy pattern generation"""
        patterns = []
        remaining_items = set(self.items)
        
        while remaining_items:
            pattern = Pattern()
            
            # Sort by dual value/area ratio for better selection
            sorted_items = sorted(remaining_items, 
                                key=lambda x: -duals.get(x.id, 0) / x.area)
            
            items_packed = []
            for item in sorted_items:
                # Try all orientations
                best_orientation = None
                for w, h in item.fits_in_bin(BIN_WIDTH, BIN_HEIGHT - pattern.height):
                    if pattern.height + h <= BIN_HEIGHT:
                        best_orientation = (w, h)
                        break
                
                if best_orientation:
                    packed_item = PackedItem(item, best_orientation[0], best_orientation[1])
                    if pattern.add_item(packed_item):
                        items_packed.append(item)
            
            if not items_packed:
                break
            
            for item in items_packed:
                remaining_items.remove(item)
            
            patterns.append(pattern)
        
        return patterns
    
    def ea_generate(self, duals: Dict[int, float]) -> Optional[Pattern]:
        """Enhanced evolutionary algorithm for pattern generation"""
        def fitness(pattern: Pattern) -> float:
            profit = pattern.total_profit(duals)
            # Add efficiency bonus to encourage better packing
            efficiency_bonus = pattern.efficiency / 100.0
            return profit + 0.1 * efficiency_bonus
        
        def create_random_pattern() -> Pattern:
            """Create a random valid pattern"""
            available_items = list(self.items)
            random.shuffle(available_items)
            return self._build_pattern_greedy(available_items, duals)
        
        def mutate(pattern: Pattern) -> Pattern:
            """Enhanced mutation with multiple strategies"""
            items_in_pattern = [pi.item for pi in pattern.items]
            if not items_in_pattern:
                return create_random_pattern()
            
            mutation_type = random.choice(['remove', 'swap', 'reorder'])
            
            if mutation_type == 'remove' and len(items_in_pattern) > 1:
                # Remove random item
                idx = random.randint(0, len(items_in_pattern) - 1)
                del items_in_pattern[idx]
            elif mutation_type == 'swap' and len(items_in_pattern) > 1:
                # Swap two items
                i, j = random.sample(range(len(items_in_pattern)), 2)
                items_in_pattern[i], items_in_pattern[j] = items_in_pattern[j], items_in_pattern[i]
            else:
                # Reorder
                random.shuffle(items_in_pattern)
            
            return self._build_pattern_greedy(items_in_pattern, duals)
        
        # Initialize population
        population = [create_random_pattern() for _ in range(EA_POP_SIZE)]
        best = max(population, key=fitness)
        no_improve = 0
        
        for gen in range(EA_MAX_GENERATIONS):
            if no_improve >= EA_MAX_NO_IMPROVE:
                break
            
            # Create offspring through mutation
            offspring = []
            for _ in range(EA_POP_SIZE // 2):
                parent = random.choice(population[:EA_POP_SIZE//2])  # Select from better half
                child = mutate(parent)
                offspring.append(child)
            
            # Combine and select best
            combined = population + offspring
            combined.sort(key=fitness, reverse=True)
            population = combined[:EA_POP_SIZE]
            
            new_best = population[0]
            if fitness(new_best) > fitness(best):
                best = new_best
                no_improve = 0
            else:
                no_improve += 1
        
        # Return pattern only if it improves the solution
        if best.total_profit(duals) > 1.0001:
            return best
        return None
    
    def _build_pattern_greedy(self, candidate_items: List[Item], duals: Dict[int, float]) -> Pattern:
        """Build pattern greedily from candidate items"""
        pattern = Pattern()
        
        # Sort by dual value for better selection
        candidate_items.sort(key=lambda x: -duals.get(x.id, 0))
        
        for item in candidate_items:
            # Try best fitting orientation
            best_fit = None
            for w, h in item.fits_in_bin(BIN_WIDTH, BIN_HEIGHT - pattern.height):
                if not best_fit or w * h > best_fit[0] * best_fit[1]:
                    best_fit = (w, h)
            
            if best_fit:
                packed_item = PackedItem(item, best_fit[0], best_fit[1])
                if not pattern.add_item(packed_item):
                    break
        
        return pattern

# ===============================================
# SECTION 5: ILP PRICING
# ===============================================

def solve_ilp_pricing(items: List[Item], duals: Dict[int, float]) -> Optional[Pattern]:
    """Solve pricing problem with enhanced ILP formulation"""
    try:
        mdl = Model(name="EnhancedPricingProblem")
        mdl.context.solver.log_output = False
        
        # Decision variables for each item and orientation
        x_vars = {}
        for item in items:
            for i, (w, h) in enumerate(item.get_orientations()):
                x_vars[(item.id, i)] = mdl.binary_var(name=f"x_{item.id}_{i}")
        
        # Constraints: at most one orientation per item
        for item in items:
            orientations = item.get_orientations()
            if len(orientations) > 1:
                mdl.add_constraint(
                    mdl.sum(x_vars[(item.id, i)] for i in range(len(orientations))) <= 1
                )
        
        # Capacity constraints
        width_constraint = mdl.sum(
            w * x_vars[(item.id, i)] 
            for item in items 
            for i, (w, h) in enumerate(item.get_orientations())
        )
        mdl.add_constraint(width_constraint <= BIN_WIDTH)
        
        height_constraint = mdl.sum(
            h * x_vars[(item.id, i)] 
            for item in items 
            for i, (w, h) in enumerate(item.get_orientations())
        )
        mdl.add_constraint(height_constraint <= BIN_HEIGHT)
        
        # Objective: maximize dual profit
        objective = mdl.sum(
            duals.get(item.id, 0) * x_vars[(item.id, i)]
            for item in items 
            for i in range(len(item.get_orientations()))
        )
        mdl.maximize(objective)
        
        # Solve
        solution = mdl.solve()
        
        if solution and solution.get_objective_value() > 1 + 1e-5:
            selected_items = []
            for item in items:
                for i, (w, h) in enumerate(item.get_orientations()):
                    if solution.get_value(x_vars[(item.id, i)]) > 0.5:
                        packed_item = PackedItem(item, w, h)
                        selected_items.append(packed_item)
                        break
            
            if selected_items:
                return Pattern(selected_items)
        
        return None
        
    except Exception as e:
        print(f"[Warning] ILP pricing failed: {e}")
        return None

# ===============================================
# SECTION 6: COLUMN GENERATION ALGORITHM
# ===============================================

def column_generation(items: List[Item]) -> Tuple[List[Pattern], ColumnGenerationTracker]:
    """Enhanced column generation with detailed tracking"""
    
    print(f"[Info] Starting column generation with {len(items)} items")
    
    generator = PatternGenerator(items)
    tracker = ColumnGenerationTracker()
    
    # Initialize with greedy patterns
    patterns = generator.greedy_generate(defaultdict(lambda: 1.0))
    print(f"[Info] Initial patterns generated: {len(patterns)}")
    
    start_time = time.time()
    
    while tracker.iteration < MAX_ITERATIONS and time.time() - start_time < MAX_TIME:
        iter_start = time.time()
        
        # Solve master problem
        try:
            mdl = Model(name=f"Master_{tracker.iteration}")
            mdl.context.solver.log_output = False
            
            # Pattern selection variables
            x = [mdl.continuous_var(lb=0, name=f"x_{i}") for i in range(len(patterns))]
            
            # Demand constraints: each item must be covered
            for item in items:
                constraint = mdl.sum(
                    x[i] for i in range(len(patterns)) 
                    if item.id in patterns[i].item_ids
                )
                mdl.add_constraint(constraint >= 1, ctname=f"demand_{item.id}")
            
            # Objective: minimize number of bins
            mdl.minimize(mdl.sum(x))
            
            solution = mdl.solve()
            
            if not solution:
                print(f"[Warning] Master problem infeasible at iteration {tracker.iteration}")
                break
            
            # Extract dual values
            duals = {}
            for i, item in enumerate(items):
                constraint = mdl.get_constraint_by_name(f"demand_{item.id}")
                duals[item.id] = constraint.dual_value if constraint.dual_value else 0
            
            obj_val = solution.get_objective_value()
            
        except Exception as e:
            print(f"[Error] Master problem failed: {e}")
            break
        
        # Solve pricing problem
        new_pattern = None
        method_used = "None"
        
        # Try EA first if enabled
        if ENABLE_EA:
            try:
                new_pattern = generator.ea_generate(duals)
                if new_pattern:
                    method_used = "EA"
            except Exception as e:
                print(f"[Warning] EA pricing failed: {e}")
        
        # Try ILP if EA failed or not enabled
        if not new_pattern and ENABLE_ILP_FALLBACK:
            try:
                new_pattern = solve_ilp_pricing(items, duals)
                if new_pattern:
                    method_used = "ILP"
            except Exception as e:
                print(f"[Warning] ILP pricing failed: {e}")
        
        iter_time = time.time() - iter_start
        
        # Check for optimality
        if not new_pattern:
            print(f"[Info] No improving pattern found at iteration {tracker.iteration}. Optimal solution reached.")
            break
        
        profit = new_pattern.total_profit(duals)
        reduced_cost = 1 - profit
        
        if profit <= 1 + 1e-5:
            print(f"[Info] Pattern profit {profit:.5f} ≤ 1. Optimal solution reached.")
            break
        
        # Add new pattern
        patterns.append(new_pattern)
        
        # Update tracker
        tracker.add_iteration(obj_val, reduced_cost, len(patterns), duals, method_used, iter_time)
        
        # Progress reporting
        if tracker.iteration % 10 == 0 or tracker.iteration <= 5:
            print(f"[Iteration {tracker.iteration:3d}] Obj: {obj_val:.3f}, RC: {reduced_cost:.4f}, "
                  f"Patterns: {len(patterns):3d}, Method: {method_used}, Time: {iter_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\n[Complete] Column generation finished:")
    print(f"  - Iterations: {tracker.iteration}")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Final patterns: {len(patterns)}")
    print(f"  - LP Objective: {tracker.objective_values[-1]:.4f}" if tracker.objective_values else "")
    
    # Solve final integer master problem
    print("\n[Final ILP] Solving for integer solution...")
    final_patterns = solve_final_ilp(items, patterns)
    
    return final_patterns, tracker

def solve_final_ilp(items: List[Item], patterns: List[Pattern]) -> List[Pattern]:
    """Solve final master problem as ILP for integer solution"""
    try:
        mdl = Model(name="FinalMasterILP")
        mdl.context.solver.log_output = False
        
        # Binary variables for pattern selection
        y = [mdl.binary_var(name=f"y_{i}") for i in range(len(patterns))]
        
        # Demand constraints
        for item in items:
            mdl.add_constraint(
                mdl.sum(y[i] for i in range(len(patterns)) if item.id in patterns[i].item_ids) >= 1
            )
        
        # Minimize number of selected patterns
        mdl.minimize(mdl.sum(y))
        
        solution = mdl.solve()
        
        if not solution:
            print("[Warning] Final ILP failed. Using LP relaxation patterns.")
            return patterns
        
        # Extract selected patterns
        selected_patterns = [
            patterns[i] for i in range(len(patterns)) 
            if solution.get_value(y[i]) > 0.5
        ]
        
        print(f"[Final ILP] Selected {len(selected_patterns)} bins from {len(patterns)} patterns")
        print(f"[Final ILP] Integer objective: {solution.get_objective_value()}")
        
        return selected_patterns
        
    except Exception as e:
        print(f"[Error] Final ILP failed: {e}")
        return patterns

# ===============================================
# SECTION 7: ADVANCED VISUALIZATION
# ===============================================

class ColumnGenerationVisualizer:
    """Advanced visualizer for column generation process"""
    
    def __init__(self, tracker_data=None):
        self.tracker = tracker_data
        plt.style.use('default')  # More compatible style
        
    def create_live_dashboard(self, save_path="column_generation_dashboard.png"):
        """Create comprehensive dashboard showing the entire CG process"""
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Main convergence plot
        ax_main = fig.add_subplot(gs[0, :2])
        ax_rc = fig.add_subplot(gs[0, 2:])
        ax_dual = fig.add_subplot(gs[1, :2])
        ax_patterns = fig.add_subplot(gs[1, 2])
        ax_methods = fig.add_subplot(gs[1, 3])
        ax_time = fig.add_subplot(gs[2, 0])
        ax_rate = fig.add_subplot(gs[2, 1])
        ax_gap = fig.add_subplot(gs[2, 2])
        ax_stats = fig.add_subplot(gs[2, 3])
        ax_stats.axis('off')
        
        if not self.tracker or not self.tracker.objective_values:
            self._create_sample_dashboard(fig, [ax_main, ax_rc, ax_dual, ax_patterns, 
                                               ax_methods, ax_time, ax_rate, ax_gap, ax_stats])
        else:
            self._populate_dashboard(fig, [ax_main, ax_rc, ax_dual, ax_patterns, 
                                          ax_methods, ax_time, ax_rate, ax_gap, ax_stats])
        
        plt.suptitle('Column Generation Process Dashboard', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _populate_dashboard(self, fig, axes):
        """Populate dashboard with real data"""
        ax_main, ax_rc, ax_dual, ax_patterns, ax_methods, ax_time, ax_rate, ax_gap, ax_stats = axes
        
        iterations = list(range(1, len(self.tracker.objective_values) + 1))
        
        # 1. Main convergence plot with dual axis
        ax_main2 = ax_main.twinx()
        
        # Objective values
        ax_main.plot(iterations, self.tracker.objective_values, 'b-o', 
                    linewidth=3, markersize=6, label='LP Objective (Bins)', alpha=0.8)
        ax_main.set_xlabel('Iteration', fontsize=12)
        ax_main.set_ylabel('Number of Bins', color='b', fontsize=12)
        ax_main.tick_params(axis='y', labelcolor='b')
        ax_main.grid(True, alpha=0.3)
        
        # Pattern count on secondary axis
        ax_main2.plot(iterations, self.tracker.pattern_counts, 'g-s', 
                     linewidth=2, markersize=4, label='Pattern Count', alpha=0.7)
        ax_main2.set_ylabel('Number of Patterns', color='g', fontsize=12)
        ax_main2.tick_params(axis='y', labelcolor='g')
        
        ax_main.set_title('Objective Value Convergence', fontsize=14, fontweight='bold')
        
        # 2. Reduced cost convergence
        ax_rc.plot(iterations, self.tracker.reduced_costs, 'r-^', 
                  linewidth=3, markersize=6, alpha=0.8)
        ax_rc.axhline(y=0, color='g', linestyle='--', linewidth=2, 
                     alpha=0.7, label='Optimality Threshold')
        ax_rc.fill_between(iterations, self.tracker.reduced_costs, 0, 
                          where=[rc > 0 for rc in self.tracker.reduced_costs], 
                          alpha=0.3, color='red', label='Improving Region')
        ax_rc.set_xlabel('Iteration', fontsize=12)
        ax_rc.set_ylabel('Reduced Cost', fontsize=12)
        ax_rc.set_title('Reduced Cost Evolution', fontsize=14, fontweight='bold')
        ax_rc.grid(True, alpha=0.3)
        ax_rc.legend()
        
        # 3. Pattern count growth
        ax_patterns.plot(iterations, self.tracker.pattern_counts, 'g-o', 
                        linewidth=2, markersize=5)
        ax_patterns.set_xlabel('Iteration', fontsize=12)
        ax_patterns.set_ylabel('Patterns', fontsize=12)
        ax_patterns.set_title('Pattern Pool Growth', fontsize=12, fontweight='bold')
        ax_patterns.grid(True, alpha=0.3)
        
        # 4. Method effectiveness pie chart
        if self.tracker.method_used:
            method_counts = pd.Series(self.tracker.method_used).value_counts()
            colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'][:len(method_counts)]
            
            ax_methods.pie(method_counts.values, labels=method_counts.index, 
                          autopct='%1.1f%%', colors=colors, startangle=90)
            ax_methods.set_title('Pattern Generation\nMethods Used', fontsize=12, fontweight='bold')
        
        # 5. Computation time per iteration
        ax_time.bar(iterations, self.tracker.computation_times, alpha=0.7, color='orange')
        ax_time.set_xlabel('Iteration', fontsize=12)
        ax_time.set_ylabel('Time (s)', fontsize=12)
        ax_time.set_title('Computation Time\nper Iteration', fontsize=12, fontweight='bold')
        ax_time.grid(True, alpha=0.3)
        
        # 6. Final statistics text
        if self.tracker.objective_values:
            stats_text = f"""
ALGORITHM PERFORMANCE

Total Iterations: {len(self.tracker.objective_values)}
Final Objective: {self.tracker.objective_values[-1]:.3f}
Final Patterns: {self.tracker.pattern_counts[-1]}

CONVERGENCE METRICS
Initial RC: {self.tracker.reduced_costs[0]:.4f}
Final RC: {self.tracker.reduced_costs[-1]:.4f}

EFFICIENCY METRICS
Total Time: {sum(self.tracker.computation_times):.2f}s
Avg Time/Iter: {np.mean(self.tracker.computation_times):.3f}s

METHOD DISTRIBUTION:
{chr(10).join([f"  {method}: {count}" for method, count in pd.Series(self.tracker.method_used).value_counts().items()])}
            """
            
            ax_stats.text(0.05, 0.95, stats_text.strip(), transform=ax_stats.transAxes, 
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    def _create_sample_dashboard(self, fig, axes):
        """Create sample dashboard when no real data is available"""
        ax_main, ax_rc, ax_dual, ax_patterns, ax_methods, ax_time, ax_rate, ax_gap, ax_stats = axes
        
        # Generate synthetic data to show dashboard layout
        iterations = list(range(1, 26))
        
        # Synthetic objective values (decreasing)
        obj_vals = [10 * np.exp(-i/15) + 3 + 0.1*np.random.random() for i in iterations]
        
        # Synthetic reduced costs (converging to 0)
        reduced_costs = [5 * np.exp(-i/8) + 0.01*np.random.random() for i in iterations]
        
        # Synthetic pattern counts (increasing)
        pattern_counts = [5 + int(i*1.2 + 0.5*np.random.random()) for i in iterations]
        
        # Sample methods
        methods = np.random.choice(['EA', 'ILP', 'Greedy'], len(iterations), p=[0.6, 0.3, 0.1])
        
        # Computation times
        comp_times = [0.1 + 0.05*np.random.random() for _ in iterations]
        
        # 1. Main convergence plot
        ax_main.plot(iterations, obj_vals, 'b-o', linewidth=3, markersize=6, alpha=0.8, label='LP Objective')
        ax_main2 = ax_main.twinx()
        ax_main2.plot(iterations, pattern_counts, 'g-s', linewidth=2, markersize=4, alpha=0.7, label='Patterns')
        ax_main.set_xlabel('Iteration')
        ax_main.set_ylabel('Number of Bins', color='b')
        ax_main2.set_ylabel('Number of Patterns', color='g')
        ax_main.set_title('Objective Value Convergence (Sample Data)', fontweight='bold')
        ax_main.grid(True, alpha=0.3)
        
        # 2. Reduced cost plot
        ax_rc.plot(iterations, reduced_costs, 'r-^', linewidth=3, markersize=6, alpha=0.8)
        ax_rc.axhline(y=0, color='g', linestyle='--', linewidth=2, alpha=0.7)
        ax_rc.set_xlabel('Iteration')
        ax_rc.set_ylabel('Reduced Cost')
        ax_rc.set_title('Reduced Cost Evolution (Sample)', fontweight='bold')
        ax_rc.grid(True, alpha=0.3)
        
        # 3. Pattern growth
        ax_patterns.plot(iterations, pattern_counts, 'g-o', linewidth=2)
        ax_patterns.set_xlabel('Iteration')
        ax_patterns.set_ylabel('Patterns')
        ax_patterns.set_title('Pattern Pool Growth (Sample)', fontweight='bold')
        ax_patterns.grid(True, alpha=0.3)
        
        # 4. Method pie chart
        method_counts = pd.Series(methods).value_counts()
        ax_methods.pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%')
        ax_methods.set_title('Methods Used (Sample)', fontweight='bold')
        
        # 5. Computation times
        ax_time.bar(iterations, comp_times, alpha=0.7, color='orange')
        ax_time.set_xlabel('Iteration')
        ax_time.set_ylabel('Time (s)')
        ax_time.set_title('Computation Time (Sample)', fontweight='bold')
        ax_time.grid(True, alpha=0.3)
        
        # 6. Sample statistics
        ax_stats.text(0.05, 0.95, """
SAMPLE DASHBOARD

This shows the layout of the
Column Generation visualizer.

Key Features:
• Real-time convergence tracking
• Dual variable evolution
• Method effectiveness analysis
• Optimality gap monitoring
• Performance metrics

Run with real data to see
actual algorithm behavior!
        """.strip(), transform=ax_stats.transAxes, fontsize=11, 
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

    def animate_convergence(self, save_path="convergence_animation.gif"):
        """Create animated visualization of convergence process"""
        if not self.tracker or not self.tracker.objective_values:
            print("No tracker data available for animation")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Show data up to current frame
            current_iterations = list(range(1, frame + 2))
            current_obj = self.tracker.objective_values[:frame + 1]
            current_rc = self.tracker.reduced_costs[:frame + 1]
            current_patterns = self.tracker.pattern_counts[:frame + 1]
            
            # Plot 1: Objective and patterns
            ax1_twin = ax1.twinx()
            
            ax1.plot(current_iterations, current_obj, 'b-o', linewidth=2, 
                    markersize=4, label='LP Objective', alpha=0.8)
            ax1_twin.plot(current_iterations, current_patterns, 'g-s', linewidth=2, 
                         markersize=4, label='Pattern Count', alpha=0.7)
            
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Number of Bins', color='b')
            ax1_twin.set_ylabel('Number of Patterns', color='g')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1_twin.tick_params(axis='y', labelcolor='g')
            ax1.set_title(f'Column Generation Progress - Iteration {frame + 1}')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Reduced cost convergence
            ax2.plot(current_iterations, current_rc, 'r-^', linewidth=2, 
                    markersize=4, alpha=0.8)
            ax2.axhline(y=0, color='g', linestyle='--', linewidth=2, alpha=0.7)
            ax2.fill_between(current_iterations, current_rc, 0, 
                           where=[rc > 0 for rc in current_rc], 
                           alpha=0.3, color='red')
            
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Reduced Cost')
            ax2.set_title('Reduced Cost (Optimality Indicator)')
            ax2.grid(True, alpha=0.3)
            
            # Add optimality status
            if current_rc[-1] <= 0.001:
                ax2.text(0.7, 0.8, 'OPTIMAL!', transform=ax2.transAxes, 
                        fontsize=14, fontweight='bold', color='green',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
            else:
                ax2.text(0.7, 0.8, f'RC: {current_rc[-1]:.4f}', transform=ax2.transAxes, 
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.8))
        
        # Create animation
        frames = len(self.tracker.objective_values)
        anim = FuncAnimation(fig, animate, frames=frames, interval=500, repeat=True)
        
        plt.tight_layout()
        
        try:
            anim.save(save_path, writer='pillow', fps=2)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Could not save animation: {e}")
            plt.show()

# ===============================================
# SECTION 8: STANDARD VISUALIZATION
# ===============================================

def visualize_bins(patterns: List[Pattern]):
    """Create detailed bin visualizations with efficiency metrics"""
    report_data = []
    efficiency_data = []
    
    for i, pattern in enumerate(patterns):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, BIN_WIDTH)
        ax.set_ylim(0, BIN_HEIGHT)
        
        efficiency = pattern.efficiency
        ax.set_title(f"Bin {i+1} - Efficiency: {efficiency:.2f}% - Area: {pattern.area}/{BIN_WIDTH*BIN_HEIGHT}", 
                    fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        
        # Add grid for better visualization
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        
        y_offset = 0
        colors = plt.cm.Set3(np.linspace(0, 1, len(pattern.items)))
        
        for j, (packed_item, color) in enumerate(zip(pattern.items, colors)):
            item, w, h = packed_item.item, packed_item.width, packed_item.height
            
            rect = Rectangle((0, y_offset), w, h, linewidth=2, 
                           edgecolor='black', facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            
            # Enhanced labeling
            rotation_text = ' (R)' if packed_item.is_rotated else ''
            ax.text(w / 2, y_offset + h / 2, 
                   f"ID {item.id}{rotation_text}\n{w}×{h}", 
                   ha='center', va='center', fontsize=9, fontweight='bold')
            
            report_data.append({
                "Bin": i+1, 
                "ItemID": item.id, 
                "Width": w, 
                "Height": h, 
                "Y_Offset": y_offset, 
                "Rotated": packed_item.is_rotated,
                "Efficiency(%)": round(efficiency, 2)
            })
            
            y_offset += h
        
        efficiency_data.append(efficiency)
        
        # Add bin boundary
        bin_rect = Rectangle((0, 0), BIN_WIDTH, BIN_HEIGHT, 
                           linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(bin_rect)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"bin_{i+1:03d}.jpg"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save detailed report
    df_report = pd.DataFrame(report_data)
    df_report.to_csv(REPORT_FILE, index=False)
    
    # Create efficiency histogram
    plt.figure(figsize=(10, 6))
    plt.hist(efficiency_data, bins=min(20, len(efficiency_data)), 
             alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Bin Packing Efficiency Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Efficiency (%)')
    plt.ylabel('Number of Bins')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_eff = np.mean(efficiency_data)
    plt.axvline(mean_eff, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_eff:.2f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "efficiency_distribution.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()

    return efficiency_data

def create_convergence_plot(tracker: ColumnGenerationTracker):
    """Create basic convergence plot"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Column Generation Convergence Analysis', fontsize=16, fontweight='bold')
    
    iterations = list(range(1, len(tracker.objective_values) + 1))
    
    # Plot 1: Objective Value Progress
    axes[0, 0].plot(iterations, tracker.objective_values, 'b-o', markersize=4, linewidth=2)
    axes[0, 0].set_title('Master Problem Objective Value')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Number of Bins')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Reduced Cost Progress
    axes[0, 1].plot(iterations, tracker.reduced_costs, 'r-s', markersize=4, linewidth=2)
    axes[0, 1].axhline(y=0, color='g', linestyle='--', alpha=0.7, label='Optimality threshold')
    axes[0, 1].set_title('Reduced Cost of New Patterns')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Reduced Cost')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Number of Patterns Growth
    axes[1, 0].plot(iterations, tracker.pattern_counts, 'g-^', markersize=4, linewidth=2)
    axes[1, 0].set_title('Pattern Pool Growth')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Number of Patterns')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Method Distribution
    if tracker.method_used:
        method_counts = pd.Series(tracker.method_used).value_counts()
        axes[1, 1].pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%',
                      colors=['lightblue', 'lightcoral', 'lightgreen'])
        axes[1, 1].set_title('Pattern Generation Methods Used')
    
    plt.tight_layout()
    plt.savefig(CONVERGENCE_PLOT, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_solution(patterns: List[Pattern], tracker: ColumnGenerationTracker):
    """Create comprehensive solution visualization"""
    
    # Main bin visualization
    efficiency_data = visualize_bins(patterns)
    
    # Convergence visualization
    create_convergence_plot(tracker)
    
    # Summary statistics
    create_summary_report(patterns, tracker, efficiency_data)

def create_summary_report(patterns: List[Pattern], tracker: ColumnGenerationTracker, efficiency_data: List[float]):
    """Create comprehensive summary report"""
    total_items = sum(len(p.items) for p in patterns)
    total_area_used = sum(p.area for p in patterns)
    total_bin_area = len(patterns) * BIN_WIDTH * BIN_HEIGHT
    overall_efficiency = (total_area_used / total_bin_area) * 100 if total_bin_area > 0 else 0
    
    summary = {
        'Metric': [
            'Total Bins Used',
            'Total Items Packed',
            'Overall Efficiency (%)',
            'Average Efficiency (%)',
            'Total Iterations',
            'Final Objective Value',
            'Total Computation Time (s)'
        ],
        'Value': [
            len(patterns),
            total_items,
            f"{overall_efficiency:.2f}",
            f"{np.mean(efficiency_data):.2f}",
            tracker.iteration,
            tracker.objective_values[-1] if tracker.objective_values else 0,
            f"{sum(tracker.computation_times):.2f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUTPUT_FOLDER, "summary_report.csv"), index=False)

# ===============================================
# SECTION 9: COMPREHENSIVE REPORTING
# ===============================================

def generate_comprehensive_report(items: List[Item], patterns: List[Pattern], tracker: ColumnGenerationTracker):
    """Generate a comprehensive text report of the analysis"""
    
    report_path = os.path.join(OUTPUT_FOLDER, "comprehensive_analysis_report.txt")
    
    # Calculate detailed statistics
    total_items = len(items)
    total_bins = len(patterns)
    total_item_area = sum(item.area for item in items)
    total_used_area = sum(pattern.area for pattern in patterns)
    total_bin_area = total_bins * BIN_WIDTH * BIN_HEIGHT
    
    overall_efficiency = (total_used_area / total_bin_area) * 100
    theoretical_min_bins = np.ceil(total_item_area / (BIN_WIDTH * BIN_HEIGHT))
    gap_from_theoretical = total_bins - theoretical_min_bins
    
    # Pattern statistics
    pattern_efficiencies = [pattern.efficiency for pattern in patterns]
    avg_efficiency = np.mean(pattern_efficiencies)
    min_efficiency = np.min(pattern_efficiencies)
    max_efficiency = np.max(pattern_efficiencies)
    
    # Items per bin statistics
    items_per_bin = [len(pattern.items) for pattern in patterns]
    avg_items_per_bin = np.mean(items_per_bin)
    
    # Algorithm performance
    total_iterations = tracker.iteration
    total_time = sum(tracker.computation_times)
    avg_time_per_iteration = np.mean(tracker.computation_times)
    
    initial_objective = tracker.objective_values[0] if tracker.objective_values else 0
    final_objective = tracker.objective_values[-1] if tracker.objective_values else 0
    objective_improvement = initial_objective - final_objective
    
    initial_rc = tracker.reduced_costs[0] if tracker.reduced_costs else 0
    final_rc = tracker.reduced_costs[-1] if tracker.reduced_costs else 0
    
    # Method analysis
    method_counts = pd.Series(tracker.method_used).value_counts() if tracker.method_used else pd.Series()
    
    # Rotation analysis
    rotated_items = sum(1 for pattern in patterns for packed_item in pattern.items if packed_item.is_rotated)
    rotation_percentage = (rotated_items / total_items) * 100 if total_items > 0 else 0
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE 2D BIN PACKING ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("PROBLEM INSTANCE\n")
        f.write("-"*20 + "\n")
        f.write(f"Input file: {INPUT_CSV}\n")
        f.write(f"Total items: {total_items}\n")
        f.write(f"Bin dimensions: {BIN_WIDTH} × {BIN_HEIGHT}\n")
        f.write(f"Bin area: {BIN_WIDTH * BIN_HEIGHT}\n")
        f.write(f"Total item area: {total_item_area}\n")
        f.write(f"Rotation allowed: {ALLOW_ROTATION}\n\n")
        
        f.write("SOLUTION QUALITY\n")
        f.write("-"*20 + "\n")
        f.write(f"Bins used: {total_bins}\n")
        f.write(f"Theoretical minimum: {theoretical_min_bins}\n")
        f.write(f"Gap from lower bound: {gap_from_theoretical} bins ({(gap_from_theoretical/theoretical_min_bins*100):.2f}%)\n")
        f.write(f"Overall efficiency: {overall_efficiency:.2f}%\n")
        f.write(f"Material waste: {100-overall_efficiency:.2f}%\n\n")
        
        f.write("BIN STATISTICS\n")
        f.write("-"*20 + "\n")
        f.write(f"Average efficiency: {avg_efficiency:.2f}%\n")
        f.write(f"Best bin efficiency: {max_efficiency:.2f}%\n")
        f.write(f"Worst bin efficiency: {min_efficiency:.2f}%\n")
        f.write(f"Efficiency std dev: {np.std(pattern_efficiencies):.2f}%\n")
        f.write(f"Average items per bin: {avg_items_per_bin:.2f}\n")
        f.write(f"Max items in a bin: {max(items_per_bin)}\n")
        f.write(f"Min items in a bin: {min(items_per_bin)}\n\n")
        
        if ALLOW_ROTATION:
            f.write("ROTATION ANALYSIS\n")
            f.write("-"*20 + "\n")
            f.write(f"Items rotated: {rotated_items} ({rotation_percentage:.2f}%)\n")
            f.write(f"Items not rotated: {total_items - rotated_items} ({100-rotation_percentage:.2f}%)\n\n")
        
        f.write("ALGORITHM PERFORMANCE\n")
        f.write("-"*20 + "\n")
        f.write(f"Total iterations: {total_iterations}\n")
        f.write(f"Total computation time: {total_time:.2f} seconds\n")
        f.write(f"Average time per iteration: {avg_time_per_iteration:.4f} seconds\n")
        f.write(f"Initial LP objective: {initial_objective:.4f}\n")
        f.write(f"Final LP objective: {final_objective:.4f}\n")
        f.write(f"LP objective improvement: {objective_improvement:.4f}\n")
        f.write(f"Initial reduced cost: {initial_rc:.6f}\n")
        f.write(f"Final reduced cost: {final_rc:.6f}\n\n")
        
        if not method_counts.empty:
            f.write("PATTERN GENERATION METHODS\n")
            f.write("-"*20 + "\n")
            for method, count in method_counts.items():
                percentage = (count / total_iterations) * 100
                f.write(f"{method}: {count} times ({percentage:.1f}%)\n")
            f.write("\n")
        
        f.write("CONVERGENCE ANALYSIS\n")
        f.write("-"*20 + "\n")
        if final_rc < 1e-6:
            f.write("Status: OPTIMAL (reduced cost < 1e-6)\n")
        elif final_rc < 1e-3:
            f.write("Status: NEAR-OPTIMAL (reduced cost < 1e-3)\n")
        else:
            f.write("Status: IMPROVING (reduced cost > 1e-3)\n")
        
        f.write(f"\nFINAL RECOMMENDATIONS\n")
        f.write("-"*20 + "\n")
        if overall_efficiency < 80:
            f.write("- Consider item sorting strategies\n")
            f.write("- Explore different bin sizes\n")
        if gap_from_theoretical > theoretical_min_bins * 0.2:
            f.write("- Large gap from theoretical minimum detected\n")
            f.write("- Consider hybrid algorithms\n")
        if rotation_percentage < 10 and ALLOW_ROTATION:
            f.write("- Low rotation usage - items may be well-sized for bins\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

def display_final_summary(items: List[Item], patterns: List[Pattern], tracker: ColumnGenerationTracker):
    """Display final summary to console"""
    total_items = len(items)
    total_bins = len(patterns)
    total_item_area = sum(item.area for item in items)
    total_used_area = sum(pattern.area for pattern in patterns)
    total_bin_area = total_bins * BIN_WIDTH * BIN_HEIGHT
    overall_efficiency = (total_used_area / total_bin_area) * 100
    theoretical_min_bins = np.ceil(total_item_area / (BIN_WIDTH * BIN_HEIGHT))
    
    print("\n" + "="*60)
    print("FINAL ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total bins used: {total_bins}")
    print(f"Theoretical minimum: {theoretical_min_bins}")
    print(f"Gap from lower bound: {total_bins - theoretical_min_bins} ({((total_bins/theoretical_min_bins - 1)*100):.1f}%)")
    print(f"Overall efficiency: {overall_efficiency:.2f}%")
    print(f"Material waste: {100-overall_efficiency:.2f}%")
    print(f"Total computation time: {sum(tracker.computation_times):.2f}s")
    print(f"Average time per iteration: {np.mean(tracker.computation_times):.3f}s")
    
    print(f"\nFiles generated in '{OUTPUT_FOLDER}':")
    print(f"  - Individual bin visualizations: bin_*.jpg")
    print(f"  - Advanced dashboard: column_generation_dashboard.png")
    print(f"  - Convergence analysis: convergence_analysis.png")
    print(f"  - Efficiency distribution: efficiency_distribution.png")
    print(f"  - Comprehensive report: comprehensive_analysis_report.txt")
    print(f"  - Summary data: summary_report.csv")
    print("="*60)

# ===============================================
# SECTION 10: MAIN INTEGRATION
# ===============================================

def run_complete_analysis():
    """
    Run the complete bin packing analysis with visualization
    This integrates the optimized algorithm with comprehensive visualization
    """
    
    print("="*80)
    print("INTEGRATED 2D BIN PACKING WITH COLUMN GENERATION ANALYSIS")
    print("="*80)
    
    try:
        # Step 1: Run the optimized bin packing algorithm
        print("\n[PHASE 1] Running Optimized Bin Packing Algorithm")
        print("-" * 50)
        
        # Load items
        items = load_items(MAX_ITEMS)
        if not items:
            raise ValueError("No items loaded. Please check your CSV file.")
        
        print(f"Loaded {len(items)} items successfully")
        
        # Run column generation with tracking
        final_patterns, tracker = column_generation(items)
        
        if not final_patterns or not tracker:
            raise Exception("Column generation failed")
        
        print(f"Column generation completed with {len(final_patterns)} final patterns")
        
        # Step 2: Create standard visualizations
        print("\n[PHASE 2] Creating Standard Solution Visualizations")
        print("-" * 50)
        
        visualize_solution(final_patterns, tracker)
        print("Standard visualizations created")
        
        # # Step 3: Create advanced column generation visualizations
        # print("\n[PHASE 3] Creating Advanced Column Generation Analysis")
        # print("-" * 50)
        
        # # Initialize the advanced visualizer with our tracking data
        # cg_visualizer = ColumnGenerationVisualizer(tracker)
        
        # # Create comprehensive dashboard
        # dashboard_path = os.path.join(OUTPUT_FOLDER, "column_generation_dashboard.png")
        # cg_visualizer.create_live_dashboard(dashboard_path)
        # print(f"Dashboard created: {dashboard_path}")
        
        # # Create animation if we have enough data
        # if len(tracker.objective_values) > 5:
        #     try:
        #         animation_path = os.path.join(OUTPUT_FOLDER, "convergence_animation.gif")
        #         cg_visualizer.animate_convergence(animation_path)
        #         print(f"Convergence animation created: {animation_path}")
        #     except Exception as e:
        #         print(f"Animation creation failed: {e}")
        
        # Step 4: Generate comprehensive report
        print("\n[PHASE 4] Generating Comprehensive Analysis Report")
        print("-" * 50)
        
        generate_comprehensive_report(items, final_patterns, tracker)
        print("Comprehensive report generated")
        
        # Step 5: Display final summary
        print("\n[PHASE 5] Final Analysis Summary")
        print("-" * 50)
        
        display_final_summary(items, final_patterns, tracker)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE - All files generated successfully!")
        print("="*80)
        
        return final_patterns, tracker
        
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main execution function"""
    try:
        # Check if input file exists
        if not os.path.exists(INPUT_CSV):
            print(f"[ERROR] Input CSV file not found: {INPUT_CSV}")
            print("Please update the INPUT_CSV path in the configuration section")
            return None, None
        
        # Display configuration
        print(f"\n[Configuration]")
        print(f"  - Input file: {INPUT_CSV}")
        print(f"  - Output folder: {OUTPUT_FOLDER}")
        print(f"  - Bin dimensions: {BIN_WIDTH} × {BIN_HEIGHT}")
        print(f"  - Max items: {MAX_ITEMS if MAX_ITEMS else 'All'}")
        print(f"  - Rotation allowed: {ALLOW_ROTATION}")
        print(f"  - EA enabled: {ENABLE_EA}")
        print(f"  - ILP fallback: {ENABLE_ILP_FALLBACK}")
        print(f"  - Max iterations: {MAX_ITERATIONS}")
        print(f"  - Max time: {MAX_TIME}s")
        
        # Run complete analysis
        return run_complete_analysis()
        
    except KeyboardInterrupt:
        print("\n[INFO] Analysis interrupted by user")
        return None, None
    except Exception as e:
        print(f"\n[ERROR] Main execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def demonstrate_sample_dashboard():
    """Create sample dashboard when no real data is available"""
    print("Creating sample dashboard to demonstrate functionality...")
    
    # Create visualizer without real data
    visualizer = ColumnGenerationVisualizer()
    
    # Create sample dashboard
    sample_path = os.path.join(OUTPUT_FOLDER, "sample_dashboard.png")
    visualizer.create_live_dashboard(sample_path)
    print(f"Sample dashboard created: {sample_path}")
    
    print("\nTo run with real data:")
    print("1. Update INPUT_CSV path to point to your data file")
    print("2. Ensure the CSV has 'Width' and 'Length' columns")
    print("3. Run main() function")

# ===============================================
# SECTION 11: EXECUTION
# ===============================================

if __name__ == "__main__":
    print("="*80)
    print("INTEGRATED 2D BIN PACKING SOLUTION")
    print("="*80)
    print("\nThis script combines:")
    print("1. Optimized 2D bin packing with column generation")
    print("2. Advanced visualization and tracking")
    print("3. Comprehensive reporting and analysis")
    
    # Check if we can run with real data
    if os.path.exists(INPUT_CSV):
        print(f"\nFound input file: {INPUT_CSV}")
        print("Running complete analysis...")
        result = main()
    else:
        print(f"\nInput file not found: {INPUT_CSV}")
        print("Creating sample dashboard instead...")
        demonstrate_sample_dashboard()
        
        print(f"\nTo run with your data:")
        print(f"1. Place your CSV file at: {INPUT_CSV}")
        print(f"2. OR update INPUT_CSV variable in the script")
        print(f"3. Ensure CSV has columns: 'Width', 'Length'")
        print(f"4. Re-run the script")

# ===============================================
# USAGE INSTRUCTIONS
# ===============================================

"""
USAGE INSTRUCTIONS:

1. SETUP:
   - Install required packages: pip install numpy pandas matplotlib seaborn docplex
   - Update INPUT_CSV path to point to your data file
   - Ensure your CSV has columns: 'Width', 'Length'

2. CONFIGURATION:
   - Adjust BIN_WIDTH, BIN_HEIGHT for your bin size
   - Set MAX_ITEMS to limit dataset size (None for all items)
   - Enable/disable ALLOW_ROTATION as needed
   - Configure algorithm parameters (EA, ILP, timeouts)

3. OUTPUT:
   The script generates:
   - Individual bin visualizations (bin_*.jpg)
   - Advanced column generation dashboard
   - Convergence analysis plots
   - Efficiency distribution analysis
   - Animated convergence (if sufficient iterations)
   - Comprehensive text report
   - CSV summary data

4. CUSTOMIZATION:
   - Modify visualization colors/styles in the plotting functions
   - Adjust algorithm parameters for your specific needs
   - Add custom reporting metrics as needed

5. TROUBLESHOOTING:
   - Check CSV file path and format
   - Ensure CPLEX/DOCPLEX is properly installed
   - Verify sufficient memory for large datasets
   - Check write permissions for OUTPUT_FOLDER

For questions or issues, check the comprehensive report generated
in the output folder for detailed algorithm performance metrics.
"""
