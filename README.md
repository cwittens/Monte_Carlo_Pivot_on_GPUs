# Monte Carlo Pivot Algorithm for Polymer Simulations

This code was written as part of a physics modeling lab course. It implements a Monte Carlo Pivot algorithm for simulating polymer chains using the pivot move method. The implementation supports both CPU and GPU backends (NVIDIA and AMD GPUs) for enhanced performance, particularly beneficial for large chain lengths where the validity check after rotation scales with $O(N^2)$. It would have probably been more efficient to implement neighbor lists, but I wanted to do some GPU programming

## Background

The pivot algorithm is a Monte Carlo method for simulating polymer chains. In each Monte Carlo step, a random monomer is chosen as a pivot point, and one segment of the chain is rotated around this pivot. The move is accepted or rejected based on the Metropolis criterion, ensuring detailed balance and proper sampling of the equilibrium distribution. In this implementation, this just means rejecting moves where individual monomers overlap (except when simulating a random walk, where every move is accepted).

## Features

- **Multi-backend support**: CPU, NVIDIA GPU (CUDA), and AMD GPU (ROCm)
- **Dynamic backend switching**: Automatically uses CPU for small chains to avoid GPU overhead
- **2D and 3D simulations**: Both off-lattice (2D) and lattice-based (3D) polymer models
- **Self-avoiding walks vs. random walks**: Compare different polymer models
- **Scaling analysis**: Automatic fitting of critical exponents $\nu$ for $R^2_{ee} ∝ N^{2\nu}$ 
- **Autocorrelation analysis**: Calculate integrated autocorrelation time
- **GPU-accelerated collision detection**: Parallel validity checking for large chains

## Project Structure

The main simulation code is contained in `mc_pivot.jl`. The project also includes:
- `Project.toml` and `Manifest.toml`: Julia package dependencies
- `plots/` directory: for output plots (created automatically)

## Setup and Configuration

### Backend Selection (Lines 36-38)
Choose your computational backend:
```julia
# backend = CUDABackend()  # for NVIDIA GPUs
# backend = ROCBackend()   # for AMD GPUs  
backend = CPU()            # for CPU only
```

### Dynamic Backend Switching (Line 42)
Enable automatic CPU fallback for small chains:
```julia
global dynamic = true  # Switch to CPU for N ≤ 100 when using GPU backend
```

### Reproducibility (Line 45)
Set random seed for reproducible results:
```julia
Random.seed!(42)
```

### Output Directory (Line 48)
Specify where to save generated plots:
```julia
plots_folder = joinpath(@__DIR__, "plots")
```

### Simulation Parameters

**Autocorrelation Analysis (Lines 429-433)**:
```julia
N_MC = 50_000         # Number of successful MC steps for autocorrelation
N = 200               # Chain length for autocorrelation analysis
t_max = 50            # Maximum lag time for autocorrelation
```

**Production Runs (Lines 430-431)**:
```julia
# All chain lengths to simulate
N_values = [10, 15, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400, 500, 700, 1000]
N_MC = 50_000           # Number of successful MC steps per chain length
```

**Equilibration Time (Line 292)**:
The warm-up period is hard-coded to 10,000 successful steps to ensure proper equilibration.

## How to Run the Code

### Prerequisites

This program was written in **Julia version 1.11.5**. Ensure you have Julia installed on your system.

### Installation and Execution

1. **Download the code**: Ensure that `mc_pivot.jl`, `Manifest.toml`, and `Project.toml` are in the same directory.

2. **Open Julia REPL**: Navigate to the project directory and start Julia:
   ```bash
   cd /path/to/project
   julia
   ```

3. **Run the simulation**:
   ```julia
   include("mc_pivot.jl")
   ```

The code will automatically:
- Calculate autocorrelation functions
- Run simulations for 2D self-avoiding walks, 2D random walks, 3D self-avoiding walks, and 3D random walks
- Generate scaling plots with fitted critical exponents
- Save all plots to the `plots/` directory

### Expected Output

The simulation generates several plots:
- `N_50000_autocorrelation_plot.pdf`: Autocorrelation function with integrated autocorrelation time
- `N_50000_2d_saw_plot.pdf`: 2D self-avoiding walk scaling (expected ν ≈ 0.75)
- `N_50000_2d_rw_plot.pdf`: 2D random walk scaling (expected ν = 0.5)
- `N_50000_3d_saw_plot.pdf`: 3D self-avoiding walk scaling (expected ν ≈ 0.588)
- `N_50000_3d_rw_plot.pdf`: 3D random walk scaling (expected ν = 0.5)









