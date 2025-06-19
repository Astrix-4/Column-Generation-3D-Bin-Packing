import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

@dataclass
class Box:
    """Represents a 3D box with dimensions and position"""
    length: float
    breadth: float
    height: float
    x: float = 0
    y: float = 0
    z: float = 0
    rotation: Tuple[int, int, int] = (0, 1, 2)  # Default: no rotation
    id: int = 0
    
    def get_rotated_dimensions(self) -> Tuple[float, float, float]:
        """Get dimensions after rotation"""
        dims = [self.length, self.breadth, self.height]
        return (dims[self.rotation[0]], dims[self.rotation[1]], dims[self.rotation[2]])
    
    def volume(self) -> float:
        return self.length * self.breadth * self.height
    
    def get_corners(self) -> List[Tuple[float, float, float]]:
        """Get all 8 corners of the box after rotation and positioning"""
        l, b, h = self.get_rotated_dimensions()
        return [
            (self.x, self.y, self.z),
            (self.x + l, self.y, self.z),
            (self.x + l, self.y + b, self.z),
            (self.x, self.y + b, self.z),
            (self.x, self.y, self.z + h),
            (self.x + l, self.y, self.z + h),
            (self.x + l, self.y + b, self.z + h),
            (self.x, self.y + b, self.z + h)
        ]

class ConstraintChecker:
    """Handles all constraint checking for 3D bin packing"""
    
    def __init__(self, bin_length: float, bin_breadth: float, bin_height: float):
        self.bin_length = bin_length
        self.bin_breadth = bin_breadth
        self.bin_height = bin_height
    
    def check_bin_boundary_constraint(self, box: Box, x: float, y: float, z: float, 
                                    rotation: Tuple[int, int, int]) -> bool:
        """Check if box fits within bin boundaries"""
        dims = [box.length, box.breadth, box.height]
        rotated_dims = (dims[rotation[0]], dims[rotation[1]], dims[rotation[2]])
        
        # Check X boundary
        if x + rotated_dims[0] > self.bin_length:
            return False
        
        # Check Y boundary
        if y + rotated_dims[1] > self.bin_breadth:
            return False
        
        # Check Z boundary
        if z + rotated_dims[2] > self.bin_height:
            return False
        
        return True
    
    def check_position_constraint(self, x: float, y: float, z: float) -> bool:
        """Check if position coordinates are valid (non-negative)"""
        return x >= 0 and y >= 0 and z >= 0
    
    def check_collision_constraint(self, box: Box, x: float, y: float, z: float,
                                 rotation: Tuple[int, int, int], placed_boxes: List[Box]) -> bool:
        """Check if box collides with any already placed boxes"""
        dims = [box.length, box.breadth, box.height]
        rotated_dims = (dims[rotation[0]], dims[rotation[1]], dims[rotation[2]])
        
        for placed_box in placed_boxes:
            if self._boxes_overlap(x, y, z, rotated_dims, placed_box):
                return False
        
        return True
    
    def _boxes_overlap(self, x1: float, y1: float, z1: float, 
                       dims1: Tuple[float, float, float], box2: Box) -> bool:
        """Check if two boxes overlap in 3D space"""
        dims2 = box2.get_rotated_dimensions()
        x2, y2, z2 = box2.x, box2.y, box2.z
        
        # Check overlap in X dimension
        if x1 >= x2 + dims2[0] or x2 >= x1 + dims1[0]:
            return False
        
        # Check overlap in Y dimension
        if y1 >= y2 + dims2[1] or y2 >= y1 + dims1[1]:
            return False
        
        # Check overlap in Z dimension
        if z1 >= z2 + dims2[2] or z2 >= z1 + dims1[2]:
            return False
        
        return True
    
    def check_stability_constraint(self, box: Box, x: float, y: float, z: float,
                                 rotation: Tuple[int, int, int], placed_boxes: List[Box]) -> bool:
        """Check if box has adequate support (not floating in air)"""
        # If box is on the ground (z=0), it's stable
        if z == 0:
            return True
        
        dims = [box.length, box.breadth, box.height]
        rotated_dims = (dims[rotation[0]], dims[rotation[1]], dims[rotation[2]])
        
        # Check if there's adequate support below
        support_area = 0
        box_base_area = rotated_dims[0] * rotated_dims[1]
        
        for placed_box in placed_boxes:
            support_area += self._calculate_support_area(x, y, z, rotated_dims, placed_box)
        
        # Require at least 50% support
        return support_area >= (box_base_area * 0.5)
    
    def _calculate_support_area(self, x: float, y: float, z: float,
                               dims: Tuple[float, float, float], support_box: Box) -> float:
        """Calculate the area of support provided by a box below"""
        support_dims = support_box.get_rotated_dimensions()
        sx, sy, sz = support_box.x, support_box.y, support_box.z
        
        # Check if support box is directly below (within tolerance)
        if abs(sz + support_dims[2] - z) > 0.01:
            return 0
        
        # Calculate overlapping area in XY plane
        overlap_x_start = max(x, sx)
        overlap_x_end = min(x + dims[0], sx + support_dims[0])
        overlap_y_start = max(y, sy)
        overlap_y_end = min(y + dims[1], sy + support_dims[1])
        
        if overlap_x_start < overlap_x_end and overlap_y_start < overlap_y_end:
            return (overlap_x_end - overlap_x_start) * (overlap_y_end - overlap_y_start)
        
        return 0
    
    def check_orientation_constraint(self, box: Box, rotation: Tuple[int, int, int]) -> bool:
        """Check if the rotation is valid (optional custom constraints)"""
        # Example: Restrict certain boxes to specific orientations
        # This can be customized based on requirements
        
        # All rotations are allowed by default
        return True
    
    def check_weight_constraint(self, box: Box, placed_boxes: List[Box]) -> bool:
        """Check weight distribution constraints (if applicable)"""
        # This is a placeholder for weight-based constraints
        # Can be implemented based on specific requirements
        return True

class RotationManager:
    """Handles rotation operations and optimization"""
    
    @staticmethod
    def get_all_rotations() -> List[Tuple[int, int, int]]:
        """Generate all possible rotations for a box"""
        return list(itertools.permutations([0, 1, 2]))
    
    @staticmethod
    def get_optimal_rotations(box: Box, target_space: Tuple[float, float, float]) -> List[Tuple[int, int, int]]:
        """Get rotations sorted by how well they fit in target space"""
        rotations = RotationManager.get_all_rotations()
        dims = [box.length, box.breadth, box.height]
        
        # Score each rotation based on fit
        scored_rotations = []
        for rotation in rotations:
            rotated_dims = (dims[rotation[0]], dims[rotation[1]], dims[rotation[2]])
            
            # Calculate fit score (lower is better)
            score = 0
            if rotated_dims[0] <= target_space[0]:
                score += (target_space[0] - rotated_dims[0])
            else:
                score += 1000  # Penalty for not fitting
            
            if rotated_dims[1] <= target_space[1]:
                score += (target_space[1] - rotated_dims[1])
            else:
                score += 1000
            
            if rotated_dims[2] <= target_space[2]:
                score += (target_space[2] - rotated_dims[2])
            else:
                score += 1000
            
            scored_rotations.append((rotation, score))
        
        # Sort by score (best fit first)
        scored_rotations.sort(key=lambda x: x[1])
        return [rotation for rotation, score in scored_rotations]

class PositionFinder:
    """Finds optimal positions for boxes"""
    
    def __init__(self, constraint_checker: ConstraintChecker):
        self.constraint_checker = constraint_checker
    
    def generate_candidate_positions(self, placed_boxes: List[Box]) -> List[Tuple[float, float, float]]:
        """Generate candidate positions for placing new boxes"""
        positions = [(0, 0, 0)]  # Origin
        
        # Add positions based on existing boxes
        for box in placed_boxes:
            dims = box.get_rotated_dimensions()
            
            # Positions adjacent to existing boxes
            positions.extend([
                (box.x + dims[0], box.y, box.z),  # Right
                (box.x, box.y + dims[1], box.z),  # Back
                (box.x, box.y, box.z + dims[2]),  # Top
                (box.x + dims[0], box.y + dims[1], box.z),  # Right-back
                (box.x + dims[0], box.y, box.z + dims[2]),  # Right-top
                (box.x, box.y + dims[1], box.z + dims[2]),  # Back-top
                (box.x + dims[0], box.y + dims[1], box.z + dims[2])  # Right-back-top
            ])
        
        # Remove duplicates and sort
        unique_positions = list(set(positions))
        
        # Sort by z (height), then y, then x for bottom-left-front filling
        unique_positions.sort(key=lambda pos: (pos[2], pos[1], pos[0]))
        
        return unique_positions
    
    def find_best_position(self, box: Box, placed_boxes: List[Box]) -> Optional[Tuple[float, float, float, Tuple[int, int, int]]]:
        """Find the best position and rotation for a box"""
        candidate_positions = self.generate_candidate_positions(placed_boxes)
        
        for position in candidate_positions:
            x, y, z = position
            
            # Try rotations in order of preference
            available_space = (
                self.constraint_checker.bin_length - x,
                self.constraint_checker.bin_breadth - y,
                self.constraint_checker.bin_height - z
            )
            
            optimal_rotations = RotationManager.get_optimal_rotations(box, available_space)
            
            for rotation in optimal_rotations:
                if self.can_place_box_at_position(box, x, y, z, rotation, placed_boxes):
                    return (x, y, z, rotation)
        
        return None
    
    def can_place_box_at_position(self, box: Box, x: float, y: float, z: float,
                                rotation: Tuple[int, int, int], placed_boxes: List[Box]) -> bool:
        """Check if box can be placed at specific position with all constraints"""
        # Check position constraint
        if not self.constraint_checker.check_position_constraint(x, y, z):
            return False
        
        # Check bin boundary constraint
        if not self.constraint_checker.check_bin_boundary_constraint(box, x, y, z, rotation):
            return False
        
        # Check collision constraint
        if not self.constraint_checker.check_collision_constraint(box, x, y, z, rotation, placed_boxes):
            return False
        
        # Check stability constraint
        if not self.constraint_checker.check_stability_constraint(box, x, y, z, rotation, placed_boxes):
            return False
        
        # Check orientation constraint
        if not self.constraint_checker.check_orientation_constraint(box, rotation):
            return False
        
        # Check weight constraint
        if not self.constraint_checker.check_weight_constraint(box, placed_boxes):
            return False
        
        return True

class SortingStrategy:
    """Handles different sorting strategies for boxes"""
    
    @staticmethod
    def sort_by_volume_descending(boxes: List[Box]) -> List[Box]:
        """Sort boxes by volume in descending order"""
        return sorted(boxes, key=lambda b: b.volume(), reverse=True)
    
    @staticmethod
    def sort_by_largest_dimension_descending(boxes: List[Box]) -> List[Box]:
        """Sort boxes by largest dimension in descending order"""
        return sorted(boxes, key=lambda b: max(b.length, b.breadth, b.height), reverse=True)
    
    @staticmethod
    def sort_by_height_descending(boxes: List[Box]) -> List[Box]:
        """Sort boxes by height in descending order"""
        return sorted(boxes, key=lambda b: b.height, reverse=True)
    
    @staticmethod
    def sort_by_aspect_ratio(boxes: List[Box]) -> List[Box]:
        """Sort boxes by aspect ratio (for better packing)"""
        def aspect_ratio_score(box):
            dims = sorted([box.length, box.breadth, box.height], reverse=True)
            return dims[0] / (dims[1] * dims[2])
        
        return sorted(boxes, key=aspect_ratio_score, reverse=True)

class WasteSpaceCalculator:
    """Calculates waste space and optimization metrics"""
    
    def __init__(self, bin_volume: float):
        self.bin_volume = bin_volume
    
    def calculate_utilization(self, placed_boxes: List[Box]) -> float:
        """Calculate space utilization percentage"""
        placed_volume = sum(box.volume() for box in placed_boxes)
        return (placed_volume / self.bin_volume) * 100
    
    def calculate_waste_space(self, placed_boxes: List[Box]) -> float:
        """Calculate waste space in cubic units"""
        placed_volume = sum(box.volume() for box in placed_boxes)
        return self.bin_volume - placed_volume
    
    def calculate_packing_efficiency(self, placed_boxes: List[Box], total_boxes: int) -> dict:
        """Calculate comprehensive packing efficiency metrics"""
        placed_volume = sum(box.volume() for box in placed_boxes)
        utilization = (placed_volume / self.bin_volume) * 100
        waste_space = self.bin_volume - placed_volume
        
        return {
            'placed_boxes': len(placed_boxes),
            'total_boxes': total_boxes,
            'placement_rate': (len(placed_boxes) / total_boxes) * 100,
            'space_utilization': utilization,
            'waste_space': waste_space,
            'placed_volume': placed_volume,
            'bin_volume': self.bin_volume
        }

class BinPacker3D:
    """Main 3D Bin Packing optimizer with modular constraints"""
    
    def __init__(self, bin_length: float, bin_breadth: float, bin_height: float):
        self.bin_length = bin_length
        self.bin_breadth = bin_breadth
        self.bin_height = bin_height
        self.bin_volume = bin_length * bin_breadth * bin_height
        
        # Initialize components
        self.constraint_checker = ConstraintChecker(bin_length, bin_breadth, bin_height)
        self.position_finder = PositionFinder(self.constraint_checker)
        self.waste_calculator = WasteSpaceCalculator(self.bin_volume)
        
        # Results
        self.placed_boxes: List[Box] = []
        self.unplaced_boxes: List[Box] = []
    
    def pack_boxes(self, boxes: List[Box], sorting_strategy: str = "volume") -> dict:
        """Pack boxes using specified sorting strategy"""
        # Apply sorting strategy
        if sorting_strategy == "volume":
            boxes_to_pack = SortingStrategy.sort_by_volume_descending(boxes)
        elif sorting_strategy == "largest_dimension":
            boxes_to_pack = SortingStrategy.sort_by_largest_dimension_descending(boxes)
        elif sorting_strategy == "height":
            boxes_to_pack = SortingStrategy.sort_by_height_descending(boxes)
        elif sorting_strategy == "aspect_ratio":
            boxes_to_pack = SortingStrategy.sort_by_aspect_ratio(boxes)
        else:
            boxes_to_pack = boxes
        
        # Add IDs to boxes for tracking
        for i, box in enumerate(boxes_to_pack):
            box.id = i + 1
        
        self.placed_boxes = []
        self.unplaced_boxes = []
        
        # Pack each box
        for box in boxes_to_pack:
            position_result = self.position_finder.find_best_position(box, self.placed_boxes)
            
            if position_result:
                x, y, z, rotation = position_result
                # Create placed box with position and rotation
                placed_box = Box(
                    length=box.length,
                    breadth=box.breadth,
                    height=box.height,
                    x=x, y=y, z=z,
                    rotation=rotation,
                    id=box.id
                )
                self.placed_boxes.append(placed_box)
            else:
                self.unplaced_boxes.append(box)
        
        # Calculate efficiency metrics
        return self.waste_calculator.calculate_packing_efficiency(
            self.placed_boxes, len(boxes)
        )
    
    def visualize_packing(self, title: str = "3D Bin Packing with Modular Constraints"):
        """Create 3D visualization of the packed boxes"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw bin boundaries
        self._draw_bin_wireframe(ax)
        
        # Generate colors for boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.placed_boxes)))
        
        # Draw each placed box
        for i, box in enumerate(self.placed_boxes):
            self._draw_box(ax, box, colors[i])
        
        # Set labels and title
        ax.set_xlabel('Length', fontsize=12)
        ax.set_ylabel('Breadth', fontsize=12)
        ax.set_zlabel('Height', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio
        max_dim = max(self.bin_length, self.bin_breadth, self.bin_height)
        ax.set_xlim(0, max_dim)
        ax.set_ylim(0, max_dim)
        ax.set_zlim(0, max_dim)
        
        # Add comprehensive legend
        efficiency = self.waste_calculator.calculate_packing_efficiency(
            self.placed_boxes, len(self.placed_boxes) + len(self.unplaced_boxes)
        )
        
        info_text = f"Packed: {efficiency['placed_boxes']}/{efficiency['total_boxes']} boxes\n"
        info_text += f"Placement Rate: {efficiency['placement_rate']:.1f}%\n"
        info_text += f"Space Utilization: {efficiency['space_utilization']:.1f}%\n"
        info_text += f"Waste Space: {efficiency['waste_space']:.1f} units³\n"
        info_text += f"Bin Volume: {efficiency['bin_volume']:.1f} units³"
        
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _draw_bin_wireframe(self, ax):
        """Draw wireframe of the bin"""
        corners = np.array([
            [0, 0, 0], [self.bin_length, 0, 0], [self.bin_length, self.bin_breadth, 0], [0, self.bin_breadth, 0],
            [0, 0, self.bin_height], [self.bin_length, 0, self.bin_height], 
            [self.bin_length, self.bin_breadth, self.bin_height], [0, self.bin_breadth, self.bin_height]
        ])
        
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # top
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical
        ]
        
        for edge in edges:
            start, end = edge
            ax.plot3D(*zip(corners[start], corners[end]), 'k--', alpha=0.6, linewidth=2)
    
    def _draw_box(self, ax, box: Box, color):
        """Draw a 3D box with labels"""
        rotated_dims = box.get_rotated_dimensions()
        l, b, h = rotated_dims
        x, y, z = box.x, box.y, box.z
        
        vertices = np.array([
            [x, y, z], [x+l, y, z], [x+l, y+b, z], [x, y+b, z],
            [x, y, z+h], [x+l, y, z+h], [x+l, y+b, z+h], [x, y+b, z+h]
        ])
        
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
            [vertices[4], vertices[7], vertices[3], vertices[0]]   # left
        ]
        
        poly3d = [[face for face in faces]]
        ax.add_collection3d(Poly3DCollection(poly3d[0], facecolors=color, linewidths=1.5, 
                                           edgecolors='black', alpha=0.7))
        
        # Add box ID label
        center_x, center_y, center_z = x + l/2, y + b/2, z + h/2
        ax.text(center_x, center_y, center_z, f'{box.id}', 
               fontsize=8, ha='center', va='center', weight='bold')

def main():
    """Main function demonstrating modular 3D bin packing with different constraints"""
    # Define bin dimensions
    bin_length, bin_breadth, bin_height = 20, 15, 12
    
    # Create 10 objects with various dimensions
    objects = [
        Box(5, 3, 4),   # Box 1
        Box(3, 3, 3),   # Box 2
        Box(4, 2, 6),   # Box 3
        Box(2, 4, 3),   # Box 4
        Box(6, 4, 2),   # Box 5
        Box(3, 5, 3),   # Box 6
        Box(4, 3, 5),   # Box 7
        Box(2, 2, 8),   # Box 8
        Box(5, 2, 3),   # Box 9
        Box(3, 4, 4),   # Box 10
    ]
    
    print("=== Modular 3D Bin Packing Optimization ===")
    print(f"Bin Dimensions: {bin_length} x {bin_breadth} x {bin_height}")
    print(f"Bin Volume: {bin_length * bin_breadth * bin_height}")
    print(f"Number of objects: {len(objects)}")
    
    print("\nObject Details:")
    total_object_volume = 0
    for i, obj in enumerate(objects, 1):
        print(f"Box {i}: {obj.length} x {obj.breadth} x {obj.height} (Volume: {obj.volume()})")
        total_object_volume += obj.volume()
    
    print(f"\nTotal object volume: {total_object_volume}")
    print(f"Theoretical max utilization: {(total_object_volume / (bin_length * bin_breadth * bin_height)) * 100:.1f}%")
    
    # Test different sorting strategies
    strategies = ["volume", "largest_dimension", "height", "aspect_ratio"]
    best_result = None
    best_strategy = None
    best_utilization = 0
    
    print("\n=== Testing Different Sorting Strategies ===")
    
    for strategy in strategies:
        packer = BinPacker3D(bin_length, bin_breadth, bin_height)
        result = packer.pack_boxes(objects.copy(), strategy)
        
        print(f"\nStrategy: {strategy.replace('_', ' ').title()}")
        print(f"  Placed: {result['placed_boxes']}/{result['total_boxes']} boxes")
        print(f"  Placement Rate: {result['placement_rate']:.1f}%")
        print(f"  Space Utilization: {result['space_utilization']:.1f}%")
        print(f"  Waste Space: {result['waste_space']:.1f} units³")
        
        if result['space_utilization'] > best_utilization:
            best_utilization = result['space_utilization']
            best_strategy = strategy
            best_result = result
            best_packer = packer
    
    print(f"\n=== Best Strategy: {best_strategy.replace('_', ' ').title()} ===")
    print(f"Space Utilization: {best_result['space_utilization']:.1f}%")
    print(f"Waste Space: {best_result['waste_space']:.1f} units³")
    
    if best_packer.unplaced_boxes:
        print(f"\nUnplaced boxes ({len(best_packer.unplaced_boxes)}):")
        for box in best_packer.unplaced_boxes:
            print(f"  Box {box.id}: {box.length} x {box.breadth} x {box.height}")
    
    print("\n=== Placed Box Details (Best Strategy) ===")
    for box in best_packer.placed_boxes:
        rotated_dims = box.get_rotated_dimensions()
        original_dims = (box.length, box.breadth, box.height)
        print(f"Box {box.id}: Position ({box.x:.1f}, {box.y:.1f}, {box.z:.1f})")
        print(f"  Original: {original_dims[0]} x {original_dims[1]} x {original_dims[2]}")
        print(f"  Rotated:  {rotated_dims[0]:.1f} x {rotated_dims[1]:.1f} x {rotated_dims[2]:.1f}")
        print(f"  Rotation: {box.rotation}")
    
    # Create visualization
    print(f"\nGenerating 3D visualization for best strategy: {best_strategy}...")
    best_packer.visualize_packing(f"Optimized 3D Bin Packing - {best_strategy.replace('_', ' ').title()} Strategy")
    
    return best_packer

if __name__ == "__main__":
    packer = main()
