# Advanced Column Generation Framework for n-D Bin Packing

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Optimization](https://img.shields.io/badge/Optimization-Linear_Programming-orange.svg)]()

## Overview
This repository contains a high-performance computational framework for solving 1D, 2D, and 3D Bin Packing Problems (BPP) using **Column Generation** and advanced **Linear Programming**. 

Initially developed to support rigorous operations research and complex constraint modeling within mechanical engineering computational pipelines, this solver utilizes a delayed column generation approach (Gilmore-Gomory formulation) to efficiently navigate massive solution spaces where standard Mixed-Integer Linear Programming (MILP) solvers hit memory bottlenecks.

## Mathematical Formulation

The architecture relies on decomposing the BPP into a **Restricted Master Problem (RMP)** and a corresponding **Pricing Problem (Subproblem)**. 

### The Restricted Master Problem (RMP)
The objective is to minimize the total number of bins utilized across the continuous relaxation of the pattern matrix:

$$\text{Minimize} \sum_{j \in \Omega} x_j$$

Subject to the demand constraints for each item type $i$:

$$\sum_{j \in \Omega} a_{ij} x_j \geq d_i \quad \forall i \in I$$

$$x_j \geq 0 \quad \forall j \in \Omega$$

Where:
* $x_j$ represents the fractional selection of cutting/packing pattern $j$.
* $a_{ij}$ is the number of items of type $i$ contained in pattern $j$.
* $d_i$ is the exact demand for item type $i$.
* $\Omega$ represents the subset of generated feasible packing patterns.

### The Pricing Problem
To dynamically generate entering variables (columns) with negative reduced costs, the subproblem solves a multidimensional knapsack problem based on the dual variables $\pi_i$ retrieved from the RMP:

$$\text{Maximize} \sum_{i \in I} \pi_i y_i$$

## Architecture & Logic
* **Iterative Generation:** Deep mathematical pattern generation is strictly implemented via iterative loops rather than recursion to ensure peak memory efficiency and prevent call-stack limits during large-scale continuous runs.
* **Memory Management:** Engineered to clear solver states between iterations, allowing it to run complex 3D spatial optimizations efficiently on unified memory architectures (tested extensively within 8GB RAM constraints).

## Dependencies
* `Python 3.9+`
* `PuLP` (Linear Programming API)
* `NumPy` / `Pandas` (Data parsing)

## Contributing
Access to the core `RMP.py` formulation is currently restricted to active institutional contributors. Please open an issue before submitting a pull request for pattern generation logic changes.
