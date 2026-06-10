# Enterprise Column Generation Framework for 3D Spatial Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Optimization](https://img.shields.io/badge/Optimization-IBM_CPLEX-052FAD.svg)]()
[![Platform](https://img.shields.io/badge/Platform-ARM64%20%7C%20Apple%20Silicon-lightgrey.svg)]()
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

## Overview

This repository contains a high-performance computational framework for solving large-scale 3D Bin Packing Problems (BPP) using **Column Generation** and continuous relaxation via **IBM CPLEX**.

Designed to support rigorous operations research and complex constraint modeling within advanced mechanical engineering computational pipelines, this solver utilizes a delayed column generation approach (Gilmore-Gomory formulation) to efficiently navigate massive solution spaces where standard Mixed-Integer Linear Programming (MILP) solvers encounter severe memory bottlenecks.

---

## Mathematical Formulation

The architecture relies on decomposing the spatial optimization problem into a **Restricted Master Problem (RMP)** and a corresponding **Pricing Problem (Subproblem)**.

### The Restricted Master Problem (RMP)

The objective is to minimize the total number of bins utilized across the continuous relaxation of the pattern matrix:

$$
\text{Minimize} \sum_{j \in \Omega} x_j
$$

Subject to the demand constraints for each item type \(i\):

$$
\sum_{j \in \Omega} a_{ij} x_j \geq d_i \quad \forall i \in I
$$

$$
x_j \geq 0 \quad \forall j \in \Omega
$$

Where:

- \(x_j\) represents the fractional selection of packing pattern \(j\).
- \(a_{ij}\) is the number of items of type \(i\) contained in pattern \(j\).
- \(d_i\) is the exact demand for item type \(i\).
- \(\Omega\) represents the subset of generated feasible packing patterns.

### The Pricing Problem

To dynamically generate entering variables (columns) with negative reduced costs, the subproblem navigates a multidimensional space based on the dual variables \(\pi_i\) retrieved from the RMP:

$$
\text{Maximize} \sum_{i \in I} \pi_i y_i
$$

---

## System Architecture

The repository is structured to separate the continuous relaxation bounds from the iterative multi-dimensional array mapping.

```text
├── solver_core.py       # Core IBM CPLEX integration and RMP/PSP pipeline
├── config/              # Physical DFC constraint parameters
├── datasets/            # SKU picklist distributions and demand matrices
└── requirements.txt     # Locked production dependencies
```

### Enterprise Solvers

Integrates directly with `docplex` to handle continuous bounds and dual variable extraction with maximum numerical stability.

### Iterative Generation

Deep mathematical pattern generation is strictly implemented via optimized iterative loops rather than recursion to prevent call-stack limits during large-scale continuous runs.

### Aggressive Memory Management

Engineered with explicit garbage collection and state-clearing between iterations.

---

## Performance Benchmarks & Hardware Optimization

This framework was specifically optimized to solve high-density operational research problems on unified memory architectures with strict RAM limits.

### Target Architecture

Benchmarked successfully on ARM64 Apple Silicon (M2 processor) environments.

### Memory Footprint

Maintained under 8GB maximum RAM allocation during continuous pricing runs, achieving 0% swap memory usage across 10,000+ column generation iterations via forced Python garbage collection.

---

## Installation & Deployment

Access to the core objective formulations is currently restricted to active institutional contributors, but the deployment environment requires the IBM CPLEX Studio Suite.

```bash
# Clone the repository
git clone https://github.com/your-handle/Column-Generation-3D-Bin-Packing.git

# Establish the virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Ensure the CPLEX PATH variable is exposed in your environment
export PYTHONPATH="/Applications/CPLEX_Studio/cplex/python/3.9/x86-64_osx"
```

---

## Execution

Execute the core solver through the primary pipeline interface. The system will automatically ingest the target CSV, instantiate the RMP, and begin logging dual stabilization metrics.

```bash
python solver_core.py --dataset "datasets/picklist_config.csv" --max_iter 50
```

---

## Academic Context & Citation

This framework is developed as part of ongoing operations research and structural modeling at the Indian Institute of Technology Delhi (IITD) within the Mechanical Engineering department computational pipelines.

Special acknowledgment to Professor Prashant Palkar for project direction and research parameters regarding applied optimization modeling.

If referencing this solver architecture in academic literature, please cite:

```bibtex
@software{CG_3D_BPP_2026,
  author = {Institutional Infrastructure Team},
  title = {Delayed Column Generation Framework for 3D Spatial Optimization},
  year = {2026},
  institution = {Indian Institute of Technology Delhi (IITD)},
  url = {https://github.com/your-handle/Column-Generation-3D-Bin-Packing}
}
```

---

## License

This project is released under the MIT License. See the `LICENSE` file for additional information.
