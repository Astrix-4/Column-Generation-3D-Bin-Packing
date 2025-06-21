import numpy as np
import pandas as pd
from scipy.optimize import linprog
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import itertools
import matplotlib.pyplot as plt
from pulp import *

@dataclass
class Box:
    """Represents a 3D box with dimensions"""
    id: int
    length: float
    breadth: float
    height: float
    
    def volume(self) -> float:
        return self.length * self.breadth * self.height
    
    def get_rotated_dimensions(self, rotation: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Get dimensions after rotation"""
        dims = [self.length, self.breadth, self.height]
        return (dims[rotation[0]], dims[rotation[1]], dims[rotation[2]])

@dataclass
class Pattern:
    """Represents a packing pattern (column in RMP)"""
    id: int
    box_assignments: Dict[int, int]  # box_id -> quantity
    utilization: float
    cost: float = 1.0  # Cost of using this pattern (typically 1 for bin minimization)
    
    def __post_init__(self):
        """Calculate total items in pattern"""
        self.total_items = sum(self.box_assignments.values())

class ConstraintManager:
    """Manages all 30 constraints for the bin packing problem"""
    
    def __init__(self, bin_length: float, bin_breadth: float, bin_height: float, boxes: List[Box]):
        self.bin_length = bin_length
        self.bin_breadth = bin_breadth
        self.bin_height = bin_height
        self.bin_volume = bin_length * bin_breadth * bin_height
        self.boxes = {box.id: box for box in boxes}
        self.constraints = self._generate_constraints()
    
    def _generate_constraints(self) -> Dict[str, Dict]:
        """Generate all 30 constraints for the problem"""
        constraints = {}
        
        # 1-10: Demand constraints (each box must be packed exactly once)
        for i, box_id in enumerate(list(self.boxes.keys())[:10], 1):
            constraints[f"demand_{i}"] = {
                'type': 'demand',
                'box_id': box_id,
                'operator': '=',
                'rhs': 1,
                'description': f"Box {box_id} must be packed exactly once"
            }
        
        # 11-15: Volume utilization constraints
        volume_thresholds = [0.6, 0.65, 0.7, 0.75, 0.8]
        for i, threshold in enumerate(volume_thresholds, 11):
            constraints[f"volume_util_{i}"] = {
                'type': 'volume_utilization',
                'threshold': threshold,
                'operator': '>=',
                'rhs': 0,
                'description': f"Volume utilization should be at least {threshold*100}%"
            }
        
        # 16-20: Dimensional constraints (max items per dimension category)
        dim_categories = [
            ('small', lambda b: max(b.length, b.breadth, b.height) <= 3),
            ('medium', lambda b: 3 < max(b.length, b.breadth, b.height) <= 6),
            ('large', lambda b: max(b.length, b.breadth, b.height) > 6),
            ('thin', lambda b: min(b.length, b.breadth, b.height) <= 2),
            ('thick', lambda b: min(b.length, b.breadth, b.height) > 4)
        ]
        
        for i, (category, condition) in enumerate(dim_categories, 16):
            constraints[f"dim_{category}_{i}"] = {
                'type': 'dimensional',
                'category': category,
                'condition': condition,
                'operator': '<=',
                'rhs': 5,
                'description': f"Max 5 {category} items per bin"
            }
        
        # 21-25: Weight distribution constraints (assuming density)
        weight_limits = [50, 60, 70, 80, 90]  # Different weight scenarios
        for i, limit in enumerate(weight_limits, 21):
            constraints[f"weight_{i}"] = {
                'type': 'weight',
                'weight_limit': limit,
                'operator': '<=',
                'rhs': limit,
                'description': f"Total weight should not exceed {limit} units"
            }
        
        # 26-28: Stability constraints
        stability_types = ['bottom_heavy', 'balanced', 'top_light']
        for i, stability in enumerate(stability_types, 26):
            constraints[f"stability_{stability}_{i}"] = {
                'type': 'stability',
                'stability_type': stability,
                'operator': '<=',
                'rhs': 1,
                'description': f"Ensure {stability} packing"
            }
        
        # 29-30: Special constraints
        constraints["fragile_29"] = {
            'type': 'fragile',
            'operator': '<=',
            'rhs': 3,
            'description': "Max 3 fragile items per bin"
        }
        
        constraints["priority_30"] = {
            'type': 'priority',
            'operator': '>=',
            'rhs': 1,
            'description': "At least 1 priority item per bin"
        }
        
        return constraints
    
    def evaluate_pattern_against_constraints(self, pattern: Pattern) -> Dict[str, float]:
        """Evaluate how a pattern performs against all constraints"""
        constraint_values = {}
        
        for constraint_name, constraint_info in self.constraints.items():
            if constraint_info['type'] == 'demand':
                box_id = constraint_info['box_id']
                constraint_values[constraint_name] = pattern.box_assignments.get(box_id, 0)
            
            elif constraint_info['type'] == 'volume_utilization':
                total_volume = sum(
                    self.boxes[box_id].volume() * qty 
                    for box_id, qty in pattern.box_assignments.items()
                )
                utilization = total_volume / self.bin_volume
                threshold = constraint_info['threshold']
                constraint_values[constraint_name] = utilization - threshold
            
            elif constraint_info['type'] == 'dimensional':
                condition = constraint_info['condition']
                count = sum(
                    qty for box_id, qty in pattern.box_assignments.items()
                    if condition(self.boxes[box_id])
                )
                constraint_values[constraint_name] = count
            
            elif constraint_info['type'] == 'weight':
                # Assume weight proportional to volume with density factor
                total_weight = sum(
                    self.boxes[box_id].volume() * qty * 2  # density factor = 2
                    for box_id, qty in pattern.box_assignments.items()
                )
                constraint_values[constraint_name] = total_weight
            
            elif constraint_info['type'] == 'stability':
                # Simplified stability metric
                stability_score = self._calculate_stability_score(pattern)
                constraint_values[constraint_name] = stability_score
            
            elif constraint_info['type'] == 'fragile':
                # Assume boxes with height > 6 are fragile
                fragile_count = sum(
                    qty for box_id, qty in pattern.box_assignments.items()
                    if self.boxes[box_id].height > 6
                )
                constraint_values[constraint_name] = fragile_count
            
            elif constraint_info['type'] == 'priority':
                # Assume boxes with volume > 50 are priority
                priority_count = sum(
                    qty for box_id, qty in pattern.box_assignments.items()
                    if self.boxes[box_id].volume() > 50
                )
                constraint_values[constraint_name] = priority_count
            
            else:
                constraint_values[constraint_name] = 0
        
        return constraint_values
    
    def _calculate_stability_score(self, pattern: Pattern) -> float:
        """Calculate stability score for a pattern"""
        if not pattern.box_assignments:
            return 0
        
        # Simple stability metric based on center of gravity
        total_weight = 0
        weighted_height = 0
        
        for box_id, qty in pattern.box_assignments.items():
            box = self.boxes[box_id]
            weight = box.volume() * qty
            total_weight += weight
            weighted_height += weight * box.height
        
        if total_weight == 0:
            return 0
        
        avg_height = weighted_height / total_weight
        return avg_height / 10  # Normalized stability score

class PatternGenerator:
    """Generates feasible packing patterns for the RMP"""
    
    def __init__(self, bin_length: float, bin_breadth: float, bin_height: float, 
                 boxes: List[Box], constraint_manager: ConstraintManager):
        self.bin_length = bin_length
        self.bin_breadth = bin_breadth
        self.bin_height = bin_height
        self.bin_volume = bin_length * bin_breadth * bin_height
        self.boxes = boxes
        self.constraint_manager = constraint_manager
    
    def generate_initial_patterns(self, num_patterns: int = 20) -> List[Pattern]:
        """Generate initial feasible patterns for RMP"""
        patterns = []
        
        # Pattern 1: Single item patterns
        for box in self.boxes[:10]:  # First 10 boxes
            pattern = Pattern(
                id=len(patterns) + 1,
                box_assignments={box.id: 1},
                utilization=box.volume() / self.bin_volume
            )
            patterns.append(pattern)
        
        # Pattern 2: Greedy volume-based patterns
        remaining_boxes = self.boxes.copy()
        for i in range(num_patterns - 10):
            pattern = self._generate_greedy_pattern(remaining_boxes, len(patterns) + 1)
            if pattern and pattern.box_assignments:
                patterns.append(pattern)
        
        return patterns
    
    def _generate_greedy_pattern(self, available_boxes: List[Box], pattern_id: int) -> Optional[Pattern]:
        """Generate a greedy pattern by filling bin with compatible boxes"""
        pattern_assignments = {}
        remaining_volume = self.bin_volume
        
        # Sort boxes by volume/space efficiency
        sorted_boxes = sorted(available_boxes, key=lambda b: b.volume(), reverse=True)
        
        for box in sorted_boxes:
            if box.volume() <= remaining_volume:
                # Check if adding this box violates constraints
                test_assignments = pattern_assignments.copy()
                test_assignments[box.id] = test_assignments.get(box.id, 0) + 1
                
                test_pattern = Pattern(
                    id=pattern_id,
                    box_assignments=test_assignments,
                    utilization=0  # Will be calculated
                )
                
                if self._is_pattern_feasible(test_pattern):
                    pattern_assignments = test_assignments
                    remaining_volume -= box.volume()
        
        if pattern_assignments:
            utilization = (self.bin_volume - remaining_volume) / self.bin_volume
            return Pattern(
                id=pattern_id,
                box_assignments=pattern_assignments,
                utilization=utilization
            )
        
        return None
    
    def _is_pattern_feasible(self, pattern: Pattern) -> bool:
        """Check if pattern satisfies basic feasibility constraints"""
        total_volume = sum(
            self.constraint_manager.boxes[box_id].volume() * qty
            for box_id, qty in pattern.box_assignments.items()
        )
        
        # Volume constraint
        if total_volume > self.bin_volume:
            return False
        
        # Basic dimensional constraints (simplified)
        for box_id, qty in pattern.box_assignments.items():
            if qty > 1:  # For simplicity, limit multiple items of same type
                return False
        
        return True
    
    def generate_column_from_dual_prices(self, dual_prices: Dict[str, float]) -> Optional[Pattern]:
        """Generate new column based on dual prices from RMP solution"""
        # This is the pricing subproblem
        # For now, implement a simplified version
        
        best_pattern = None
        best_reduced_cost = float('inf')
        
        # Try different combinations of boxes
        for r in range(1, min(4, len(self.boxes) + 1)):  # Up to 3 boxes per pattern
            for box_combination in itertools.combinations(self.boxes, r):
                pattern_assignments = {box.id: 1 for box in box_combination}
                
                pattern = Pattern(
                    id=999,  # Temporary ID
                    box_assignments=pattern_assignments,
                    utilization=sum(box.volume() for box in box_combination) / self.bin_volume
                )
                
                if self._is_pattern_feasible(pattern):
                    reduced_cost = self._calculate_reduced_cost(pattern, dual_prices)
                    
                    if reduced_cost < best_reduced_cost:
                        best_reduced_cost = reduced_cost
                        best_pattern = pattern
        
        # Return pattern only if it has negative reduced cost
        if best_reduced_cost < -1e-6:
            return best_pattern
        
        return None
    
    def _calculate_reduced_cost(self, pattern: Pattern, dual_prices: Dict[str, float]) -> float:
        """Calculate reduced cost for a pattern"""
        # Start with pattern cost (typically 1 for bin minimization)
        reduced_cost = pattern.cost
        
        # Subtract dual prices for constraints this pattern satisfies
        constraint_values = self.constraint_manager.evaluate_pattern_against_constraints(pattern)
        
        for constraint_name, value in constraint_values.items():
            if constraint_name in dual_prices:
                reduced_cost -= dual_prices[constraint_name] * value
        
        return reduced_cost

class RestrictedMasterProblem:
    """Solves the Restricted Master Problem using linear programming"""
    
    def __init__(self, boxes: List[Box], constraint_manager: ConstraintManager):
        self.boxes = boxes
        self.constraint_manager = constraint_manager
        self.patterns: List[Pattern] = []
        self.solution = None
        self.dual_prices = {}
        self.objective_value = float('inf')
    
    def add_patterns(self, new_patterns: List[Pattern]):
        """Add new patterns to the RMP"""
        for pattern in new_patterns:
            pattern.id = len(self.patterns) + 1
            self.patterns.append(pattern)
    
    def solve_rmp(self) -> Dict:
        """Solve the current RMP using PuLP"""
        if not self.patterns:
            return {'status': 'No patterns available'}
        
        # Create the linear programming problem
        prob = LpProblem("Bin_Packing_RMP", LpMinimize)
        
        # Decision variables: x_j for each pattern j
        pattern_vars = {}
        for pattern in self.patterns:
            pattern_vars[pattern.id] = LpVariable(f"x_{pattern.id}", lowBound=0, cat='Continuous')
        
        # Objective function: minimize number of bins
        prob += lpSum([pattern.cost * pattern_vars[pattern.id] for pattern in self.patterns])
        
        # Add all 30 constraints
        for constraint_name, constraint_info in self.constraint_manager.constraints.items():
            constraint_expr = 0
            
            for pattern in self.patterns:
                constraint_values = self.constraint_manager.evaluate_pattern_against_constraints(pattern)
                coefficient = constraint_values.get(constraint_name, 0)
                constraint_expr += coefficient * pattern_vars[pattern.id]
            
            # Add constraint based on operator
            if constraint_info['operator'] == '=':
                prob += constraint_expr == constraint_info['rhs'], constraint_name
            elif constraint_info['operator'] == '<=':
                prob += constraint_expr <= constraint_info['rhs'], constraint_name
            elif constraint_info['operator'] == '>=':
                prob += constraint_expr >= constraint_info['rhs'], constraint_name
        
        # Solve the problem
        prob.solve(PULP_CBC_CMD(msg=0))
        
        # Extract results
        if prob.status == 1:  # Optimal
            self.solution = {}
            self.dual_prices = {}
            
            # Extract primal solution
            for pattern in self.patterns:
                var_value = pattern_vars[pattern.id].varValue
                if var_value and var_value > 1e-6:
                    self.solution[pattern.id] = var_value
            
            # Extract dual prices
            for constraint_name in self.constraint_manager.constraints.keys():
                for constraint in prob.constraints:
                    if constraint.name == constraint_name:
                        self.dual_prices[constraint_name] = constraint.pi if constraint.pi else 0
                        break
            
            self.objective_value = value(prob.objective)
            
            return {
                'status': 'Optimal',
                'objective_value': self.objective_value,
                'solution': self.solution,
                'dual_prices': self.dual_prices,
                'num_patterns': len(self.patterns),
                'active_patterns': len(self.solution)
            }
        
        else:
            return {
                'status': 'Infeasible or Unbounded',
                'objective_value': None
            }
    
    def print_solution_summary(self):
        """Print detailed solution summary"""
        if not self.solution:
            print("No solution available")
            return
        
        print("\n=== RMP Solution Summary ===")
        print(f"Objective Value (Number of Bins): {self.objective_value:.4f}")
        print(f"Total Patterns: {len(self.patterns)}")
        print(f"Active Patterns: {len(self.solution)}")
        
        print("\n=== Active Patterns ===")
        for pattern_id, value in self.solution.items():
            pattern = next(p for p in self.patterns if p.id == pattern_id)
            print(f"Pattern {pattern_id}: {value:.4f}")
            print(f"  Box assignments: {pattern.box_assignments}")
            print(f"  Utilization: {pattern.utilization:.2%}")
        
        print("\n=== Dual Prices (Shadow Prices) ===")
        for constraint_name, dual_price in self.dual_prices.items():
            if abs(dual_price) > 1e-6:
                constraint_info = self.constraint_manager.constraints[constraint_name]
                print(f"{constraint_name}: {dual_price:.4f} - {constraint_info['description']}")
    
    def visualize_solution(self):
        """Visualize the RMP solution"""
        if not self.solution:
            print("No solution to visualize")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Pattern usage
        pattern_ids = list(self.solution.keys())
        pattern_values = list(self.solution.values())
        
        ax1.bar(range(len(pattern_ids)), pattern_values)
        ax1.set_xlabel('Pattern ID')
        ax1.set_ylabel('Usage (Fractional)')
        ax1.set_title('Pattern Usage in RMP Solution')
        ax1.set_xticks(range(len(pattern_ids)))
        ax1.set_xticklabels(pattern_ids)
        
        # Plot 2: Constraint dual prices
        significant_duals = {k: v for k, v in self.dual_prices.items() if abs(v) > 1e-6}
        
        if significant_duals:
            constraint_names = list(significant_duals.keys())
            dual_values = list(significant_duals.values())
            
            ax2.barh(range(len(constraint_names)), dual_values)
            ax2.set_ylabel('Constraint')
            ax2.set_xlabel('Dual Price')
            ax2.set_title('Significant Dual Prices')
            ax2.set_yticks(range(len(constraint_names)))
            ax2.set_yticklabels([name.replace('_', ' ') for name in constraint_names])
        else:
            ax2.text(0.5, 0.5, 'No significant dual prices', 
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('Dual Prices')
        
        plt.tight_layout()
        plt.show()

class ColumnGenerationSolver:
    """Main column generation solver"""
    
    def __init__(self, bin_length: float, bin_breadth: float, bin_height: float, boxes: List[Box]):
        self.bin_length = bin_length
        self.bin_breadth = bin_breadth
        self.bin_height = bin_height
        self.boxes = boxes
        
        # Initialize components
        self.constraint_manager = ConstraintManager(bin_length, bin_breadth, bin_height, boxes)
        self.pattern_generator = PatternGenerator(bin_length, bin_breadth, bin_height, boxes, self.constraint_manager)
        self.rmp = RestrictedMasterProblem(boxes, self.constraint_manager)
        
        self.iteration_log = []
    
    def solve(self, max_iterations: int = 20) -> Dict:
        """Solve using column generation"""
        print("=== Starting Column Generation ===")
        
        # Generate initial patterns
        initial_patterns = self.pattern_generator.generate_initial_patterns()
        self.rmp.add_patterns(initial_patterns)
        
        print(f"Generated {len(initial_patterns)} initial patterns")
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Solve current RMP
            rmp_result = self.rmp.solve_rmp()
            
            if rmp_result['status'] != 'Optimal':
                print(f"RMP Status: {rmp_result['status']}")
                break
            
            print(f"RMP Objective: {rmp_result['objective_value']:.4f}")
            print(f"Active patterns: {rmp_result['active_patterns']}/{rmp_result['num_patterns']}")
            
            # Log iteration
            self.iteration_log.append({
                'iteration': iteration,
                'objective': rmp_result['objective_value'],
                'num_patterns': rmp_result['num_patterns'],
                'active_patterns': rmp_result['active_patterns']
            })
            
            # Generate new column using dual prices
            new_pattern = self.pattern_generator.generate_column_from_dual_prices(self.rmp.dual_prices)
            
            if new_pattern is None:
                print("No improving column found. Optimal solution reached.")
                break
            
            # Add new pattern to RMP
            self.rmp.add_patterns([new_pattern])
            print(f"Added new pattern with {len(new_pattern.box_assignments)} boxes")
        
        # Final solution
        final_result = {
            'status': 'Optimal' if iteration < max_iterations else 'Max iterations reached',
            'iterations': iteration,
            'final_objective': self.rmp.objective_value,
            'total_patterns': len(self.rmp.patterns),
            'solution': self.rmp.solution,
            'dual_prices': self.rmp.dual_prices
        }
        
        print(f"\n=== Column Generation Complete ===")
        print(f"Status: {final_result['status']}")
        print(f"Iterations: {final_result['iterations']}")
        print(f"Final Objective: {final_result['final_objective']:.4f}")
        
        return final_result
    
    def plot_convergence(self):
        """Plot convergence of column generation"""
        if not self.iteration_log:
            print("No iteration data available")
            return
        
        iterations = [log['iteration'] for log in self.iteration_log]
        objectives = [log['objective'] for log in self.iteration_log]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, objectives, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value (Number of Bins)')
        plt.title('Column Generation Convergence')
        plt.grid(True, alpha=0.3)
        plt.show()

def main():
    """Main function to demonstrate column generation for 3D bin packing"""
    
    # Define bin and boxes
    bin_length, bin_breadth, bin_height = 20, 15, 12
    
    boxes = [
        Box(1, 5, 3, 4),   Box(2, 3, 3, 3),   Box(3, 4, 2, 6),   Box(4, 2, 4, 3),   Box(5, 6, 4, 2),
        Box(6, 3, 5, 3),   Box(7, 4, 3, 5),   Box(8, 2, 2, 8),   Box(9, 5, 2, 3),   Box(10, 3, 4, 4)
    ]
    
    print("=== 3D Bin Packing with Column Generation ===")
    print(f"Bin dimensions: {bin_length} x {bin_breadth} x {bin_height}")
    print(f"Number of boxes: {len(boxes)}")
    print(f"Number of constraints: 30")
    
    # Initialize solver
    solver = ColumnGenerationSolver(bin_length, bin_breadth, bin_height, boxes)
    
    # Print constraint summary
    print("\n=== Constraint Summary ===")
    for i, (name, info) in enumerate(solver.constraint_manager.constraints.items(), 1):
        print(f"{i:2d}. {name}: {info['description']}")
    
    # Solve using column generation
    result = solver.solve(max_iterations=10)
    
    # Print detailed results
    solver.rmp.print_solution_summary()
    
    # Visualize results
    solver.rmp.visualize_solution()
    solver.plot_convergence()
    
    return solver

if __name__ == "__main__":
    solver = main()
