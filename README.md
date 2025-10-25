# Island-based Ant Colony Optimization with Master Colony (IACOMC)

A novel hybrid algorithm for solving the Capacitated Vehicle Routing Problem (CVRP) using an innovative Master Colony mechanism combined with island-based parallel optimization.



## Academic Project

This repository contains the implementation of my undergraduate dissertation research at the University of Birmingham (2023-24), supervised by Amina Alkazemi. The work addresses fundamental limitations in traditional Ant Colony Optimization algorithms when solving vehicle routing problems.

## Table of Contents

- [Problem Statement](#problem-statement)
- [The Solution](#the-solution)
- [Key Innovations](#key-innovations)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Architecture](#algorithm-architecture)
- [Performance](#performance)

## Problem Statement

The **Capacitated Vehicle Routing Problem (CVRP)** is a critical NP-hard optimization challenge faced by logistics companies like Amazon. The problem involves:

- Finding optimal routes for a fleet of vehicles
- Delivering goods from a central depot to multiple customers
- Respecting vehicle capacity constraints
- Minimizing total travel distance/cost

### Limitations of Traditional Approaches

Traditional Ant Colony Optimization (ACO) and Max-Min Ant System (MMAS) algorithms suffer from three major issues:

1. **Poor solution quality** - Often failing to find optimal or near-optimal solutions
2. **Slow convergence speed** - Requiring excessive computational time
3. **Premature convergence** - Getting trapped in local optima

## The Solution

IACOMC introduces a **Master Colony mechanism** that:

- Combines pheromone matrices from multiple independent ant colonies
- Leverages the strengths of island-model parallel computing
- Restricts unpromising routes while amplifying successful paths
- Integrates local search heuristics (2-opt and inter-swap) for refinement

Think of it as a "wisdom of crowds" approach - multiple ant colonies explore the solution space independently, and the Master Colony synthesizes their collective knowledge to guide the search toward better solutions.

## Key Innovations

### 1. Master Colony Mechanism

The core innovation combines multiple pheromone matrices using weighted aggregation:

```python
# Weighted aggregation of colony pheromone matrices
Master_Matrix = Σ(Colony_Matrix_i × Weight_i)
```

Where weights are based on:
- Colony fitness rankings
- Normalized pheromone values
- Configurable penalty parameter γ

### 2. Pseudo Island

A supporting colony with low evaporation rate (ρ = 0.58) that:
- Maintains a broader solution space
- Prevents "dead-end" scenarios in the Master Colony
- Expands path intersections between colonies

### 3. Strategic Migration

- **Synchronous migration** every 400 iterations
- Fully connected topology for rapid information spread
- Controlled pheromone matrix updates to prevent overwhelming

### 4. Hybrid Local Search

- **2-opt**: Eliminates route crossings
- **Inter-swap**: Exchanges customers between vehicles
- Adaptive limits that boost during Master Colony re-initialization

## Results

Tested on Augerat benchmark datasets (Set A), IACOMC demonstrated:

| Instance | BKS | IACOMC+LS | Traditional ACO | Gap (%) | Improvement |
|----------|-----|-----------|-----------------|---------|-------------|
| A-n33-k5 | 661 | **661*** | 675 | 0.00 | **Optimal** |
| A-n33-k6 | 742 | **742*** | 776 | 0.00 | **Optimal** |
| A-n48-k7 | 1073 | **1073*** | 1220 | 0.00 | **Optimal** |
| A-n60-k9 | 1354 | **1354*** | 1470 | 0.00 | **Optimal** |
| A-n69-k9 | 1159 | **1159*** | 1234 | 0.00 | **Optimal** |
| A-n80-k10 | 1763 | 1779 | 1997 | 0.91 | **10.9% better** |

*Achieved optimal solution

### Statistical Validation

**Wilcoxon signed-rank test** (30 runs per instance):
- **p < 0.001** for all instances vs. traditional ACO
- **p < 0.05** for all instances vs. MMAS
- Demonstrates statistically significant superiority

### Key Performance Metrics

- **3-5x faster** convergence to near-optimal solutions
- **100% success rate** finding BKS on small-medium instances (33-69 customers)
- **<1% gap** on larger instances (80 customers)
- Reduced premature convergence through continuous Master Colony re-initialization

## Installation

### Requirements

- Python 3.10.14+
- Numba 0.60.0 (for JIT compilation and parallelization)
- VRPLib (for dataset handling)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/Alexandercheng-rsch/ACO.git
cd ACO

# Install dependencies
pip install -r requirements.txt
```

### System Requirements

The algorithm is computationally intensive. Tested on:
- **CPU**: AMD Ryzen 9 6900HX (3.3 GHz)
- **RAM**: 32 GB
- **GPU**: RTX 3070 Ti (optional, for Numba acceleration)

**Note**: Can run on modest hardware but will be slower. Parallelization through Numba provides significant speedup.

## Usage

### Basic Usage

1. **Select an instance** from the `instances/` directory
2. **Edit configuration** in `ACO_VRP_ISLAND.py`:

```python
# Example configuration
file = "instances/A-n33-k5.vrp"  # Instance file path
drivers = 5                       # Number of vehicles (k value from filename)
```

3. **Run the algorithm**:

```bash
python ACO_VRP_ISLAND.py
```

### Configuration Parameters

Key parameters in the algorithm:

```python
# Master Colony parameters
alpha_mc = 1.1      # Pheromone influence
beta_mc = 0.89      # Heuristic influence
rho_mc = 0.03       # Evaporation rate
gamma = 2.6         # Penalty parameter for colony weighting

# Worker Colony parameters (MMAS variant)
alpha = 1.1         # Pheromone influence
beta = 2.7          # Heuristic influence
rho = 0.99          # Evaporation rate (high for exploitation)
p_best = 0.05       # MMAS convergence control

# Pseudo Island parameters
alpha_pseudo = 1.0
beta_pseudo = 2.7
rho_pseudo = 0.58   # Low evaporation for broader search
p_best_pseudo = 0.85

# System parameters
num_ants = 300
num_worker_islands = 8
migration_rate = 400  # Iterations between migrations
```

### Understanding the Parameters

- **α (alpha)**: Controls pheromone influence. Higher = more reliance on learned paths
- **β (beta)**: Controls heuristic influence. Higher = more greedy (prefer shorter distances)
- **ρ (rho)**: Evaporation rate. Higher = faster forgetting of old information
- **γ (gamma)**: Penalizes lower-ranked colonies in Master Colony aggregation
- **p_best**: MMAS parameter controlling exploitation vs. exploration

### Output

The algorithm outputs:
- Best solution found (route assignments per vehicle)
- Total distance/cost
- Convergence graphs showing fitness over iterations
- Statistical summaries over 30 runs (mean, std, min, max)

## Algorithm Architecture

### Island Model Structure

```
┌─────────────────────────────────────────────┐
│           Master Colony                     │
│   (Aggregates pheromone matrices)           │
│   • Traditional ACO variant                 │
│   • Uses weighted colony matrices           │
│   • Re-initialized every migration period   │
└───────────┬─────────────────────────────────┘
            │ shares best solutions
            ▼
┌───────────────────────────────────────────────┐
│          Worker Colonies (8 islands)          │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐           │
│  │ C1  │◄─┤ C2  │◄─┤ C3  │◄─┤ C4  │           │
│  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘           │
│     │        │        │        │              │
│  ┌──▼──┐  ┌─▼───┐  ┌─▼───┐  ┌─▼───┐           │
│  │ C5  │◄─┤ C6  │◄─┤ C7  │◄─┤ C8  │           │
│  └─────┘  └─────┘  └─────┘  └─────┘           │
│         (MMAS variant)                        │
│   • Independent evolution                     │
│   • Pheromone bounds (τ_min, τ_max)           │
│   • Fully connected topology                  │
└───────────────────────────────────────────────┘
            ▲
            │ receives solutions (one-way)
┌───────────┴─────────────────────────────────┐
│          Pseudo Island                      │
│   • Low evaporation rate (ρ = 0.58)         │
│   • Maintains broad solution space          │
│   • Prevents Master Colony dead-ends        │
└─────────────────────────────────────────────┘
```

### Detailed Workflow Steps

1. **Initialization**: Each colony starts with uniform pheromone trails (τ = 1)
2. **Solution Construction**: 300 ants per colony build routes using probabilistic transition rules
3. **Local Search**: 
   - 2-opt eliminates route crossings
   - Inter-swap exchanges customers between vehicles
4. **Pheromone Update**: MMAS-style updates with dynamic bounds (τ_min, τ_max)
5. **Migration** (every 400 iterations):
   - Worker colonies share best solutions (fully connected)
   - Master Colony aggregates all pheromone matrices
   - Pseudo island provides stability
6. **Master Colony Construction**: Generates solutions using combined knowledge
7. **Boost Phase**: Temporarily increase local search limits for 5 iterations
8. **Repeat** until convergence or maximum iterations

## Performance Optimization

### Parallelization Strategy

The implementation uses **Numba JIT compilation** for:
- Parallel ant solution construction (300 ants simultaneously)
- Vectorized distance calculations
- Accelerated local search operations (2-opt, inter-swap)

```python
@njit(parallel=True)
def construct_solutions(num_ants, pheromone_matrix, distance_matrix):
    # Parallelized across ants
    for ant in prange(num_ants):
        # Each ant constructs solution independently
        ...
```

This enables **single-machine execution** of what would typically require distributed computing across multiple servers.

### Performance Characteristics

| Instance Size | Avg Time per Iteration | 30 Runs Total Time |
|--------------|------------------------|-------------------|
| 33 customers | ~2 seconds | ~15 minutes |
| 60 customers | ~5 seconds | ~40 minutes |
| 80 customers | ~8 seconds | ~65 minutes |

*Times measured on AMD Ryzen 9 6900HX with RTX 3070 Ti*

## Convergence Analysis

### Key Observations

- **Initial Boost**: Master Colony initialization provides immediate 5-10% improvement
- **Sustained Progress**: Continues improving during worker colony stagnation periods
- **Stability**: Re-initialization every 400 iterations prevents premature convergence
- **Scalability**: Better relative performance on medium-to-large instances (60-80 customers)

### Example Convergence (A-n69-k9)

```
Iteration    ACO     MMAS    IACOMC-LS    IACOMC+LS
   400      1284     1394      1225         1159*  ← Master Colony init
   800      1267     1388      1210         1159*
  1200      1259     1388      1198         1159*  ← Re-initialization
  1600      1251     1382      1189         1159*
  2000      1245     1380      1181         1159*  ← Optimal reached
```

## Research Contributions

1. **Novel Master Colony Mechanism**: First application of weighted pheromone matrix aggregation in ACO-based island models
2. **Pseudo Island Concept**: Innovative solution to prevent solution space deadlocks in sparse combined matrices
3. **Empirical Validation**: Comprehensive statistical testing on benchmark instances with 30 runs per configuration
4. **Practical Implementation**: Single-machine parallelization using Numba (no distributed system required)
5. **Open Source**: Full implementation and dissertation available for reproducibility

## Repository Structure

```
ACO/
├── ACO_VRP_ISLAND.py           # Main IACOMC algorithm implementation
├── path_construction.py         # Route construction utilities & heuristics
├── requirements.txt             # Python dependencies
├── Generic_Report_Template.pdf  # Full dissertation (60 pages)
├── instances/                   # Augerat CVRP benchmark datasets
│   ├── A-n33-k5.vrp
│   ├── A-n46-k7.vrp
│   ├── A-n60-k9.vrp
│   ├── A-n69-k9.vrp
│   ├── A-n80-k10.vrp
│   └── ...
└── README.md
```

## Future Work

Potential extensions and improvements:

- **Genetic Algorithm Integration**: Two-level archipelago with GA + ACO
- **Alternative ACO Variants**: Test Ant Colony System (ACS) for worker colonies
- **Normalization Techniques**: Experiment with Z-score, softmax, etc.
- **Larger Instances**: Scale to 100+ customers (requires parameter tuning)
- **Dynamic CVRP**: Real-time logistics with time windows and traffic
- **Multi-depot VRP**: Extend Master Colony to multi-depot scenarios
- **Hyperparameter Optimization**: Deep learning-based parameter adaptation

## Related Work

This research builds upon:

- **Max-Min Ant System** (Stützle & Hoos, 1997)
- **Island Models in Evolutionary Computing** (Dorigo & Di Caro, 1999)
- **Hybrid ACO for CVRP** (Gao & Wu, 2023; Cai & Wang, 2022)

Key differentiator: First to combine **island-based ACO with pheromone matrix aggregation** specifically for CVRP.

## Acknowledgements
- **Benchmark Datasets**: Augerat et al. (1995)
- **Theoretical Foundation**: Max-Min Ant System (Stützle & Hoos, 1997)
- **Optimization Framework**: Optuna (for hyperparameter tuning)

## Contact

For questions about the algorithm, implementation, or potential collaboration, please open an issue or reach out via GitHub.

---

## Key Achievements

- **5/9 optimal solutions** found on benchmark instances
- **100% statistical significance** vs. baseline algorithms
- **3-5x faster convergence** than traditional ACO/MMAS
- **Novel mechanism** with potential applications beyond CVRP

---
