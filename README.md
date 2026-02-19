# NucleonLCCQM.jl

Julia package for computing observables in a light-cone constituent quark model.

## About
We parametrize the light-cone baryon wavefunction in a truncated three-quark Fock sector, compute the electromagnetic Dirac/Pauli form factors, build the cubic color correlator, and integrate it to obtain the gluon Sivers function at non-vanishing transverse momentum transfer.

## Installation

From the Julia REPL:
```julia
using Pkg
Pkg.add(url="https://github.com/HechenMountain/nucleon-lccqm")
```

Or for development:
```julia
Pkg.develop(url="https://github.com/HechenMountain/nucleon-lccqm")
```

## Setup (after cloning)

Requires **Julia 1.9+** (tested on Julia 1.12).

```bash
cd nucleon-lccqm
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

This reads `Project.toml` and downloads/precompiles all Julia dependencies (`Cuba`, `SpecialFunctions`, `StaticArrays`, etc.).

Verify the install:
```julia
julia --project=. -e "using NucleonLCCQM; println(NucleonLCCQM.MQ)"
# Expected output: 0.28 (MQ from parameters.jl)
```

## Features
- Multiple wavefunction parametrizations selectable in `src/parameters.jl` (`WF_TYPE = :pow` or `:exp`)
- Parallelized data generation for form factors, odderon distributions, and the Sivers function using writers.jl script
- Symbol-based configuration for solvers (`:cuhre`, `:vegas`, `:suave`, `:divonne`)
- Linear/log spacing controls for writers (`spacing=:lin` or `:log`)
- `StaticArrays`-based 2D vector helpers for lower-allocation hot paths (APIs accept `AbstractVector`, including `Vector` and `SVector`)

## Quick start

```julia
using NucleonLCCQM
using StaticArrays

# Example: gluon Sivers at [kx,ky]=[0.5,0] GeV with vegas
res, err, prob, neval, fail, nregions = gluon_sivers([0.5,0]; μ=0.0, solver=:vegas)

# SVector inputs are also supported
res_s, err_s, prob_s, neval_s, fail_s, nregions_s = gluon_sivers(@SVector [0.5, 0.0]; μ=0.0, solver=:vegas)

# Form factors
f1, err_f1, prob, neval, fail, nregions = f1_form_factor([0.1, 0.0])
f2, err_f2, prob, neval, fail, nregions = f2_form_factor([0.1, 0.0])

# Normalize wavefunction
norm, err, prob, neval, fail, nregions = normalize_wavefunction()
```

## Batch runs
Use the writer script with 4 parallel workers from the project root:
```sh
julia --project=. -p 4 scripts/writers.jl
```
Or on Linux/macOS, to run in the background:
```sh
nohup julia --project=. -p 4 scripts/writers.jl > writers.log 2>&1 &
```
Uncomment or edit the desired `write_*` calls near the end of `scripts/writers.jl`. All solver arguments are Symbols (e.g., `solver=:vegas`), and spacing is controlled with `spacing=:lin` or `:log`.

## Notebooks
See `notebooks/sivers.ipynb` for interactive examples.

### Plot Generation
Use `notebooks/plots.ipynb` to generate figures from generated data:

**Available plots:**
- Form factors (F₁/F₂) with experimental and power-law parametrization comparisons
- Gluon Sivers function with log-fit analysis
- Spin-dependent odderon distribution with fitted model curves
- Real-space and momentum-space evolution distributions (Y = 0,1,2,4) (Evolution done separately)
- Fourier-transform comparisons
- Open-charm production cross-sections

**Data Organization:**
- `data/csv/exp/` — data from experimental wavefunction parametrization
- `data/csv/pow/` — data from power-law wavefunction parametrization  
- `data/csv/1707.09063/` — reference form factor data

**Requirements:**
- Python 3.7+
- **Core packages:**
  - `numpy` — numerical computing
  - `scipy` — integration, interpolation, optimization, special functions
  - `matplotlib` — plotting
  - `joblib` — parallel execution
- **Optional:**
  - LaTeX

**Usage:**
1. Generate data using `scripts/writers.jl`
2. Open `notebooks/plots.ipynb` in Jupyter and run cells sequentially
3. PDFs are saved to `data/plots/exp/` and `data/plots/pow/`, respectively

## Configuration
Adjust model parameters in `src/parameters.jl`:
- `WF_TYPE`: wavefunction type (`:pow` or `:exp`)
- `MQ`: constituent quark mass
- `BETA`: baryon wavefunction width parameter
- `NORM`: wavefunction normalization (from `normalize_wavefunction()`)
- `NC`: number of colors
- `ALPHA_S`: strong coupling constant
- `M_N`: nucleon mass

## Project Structure
```
nucleon-lccqm/
├── Project.toml            # Package metadata & dependencies
├── src/
│   ├── NucleonLCCQM.jl     # Main module (entry point, includes all submodules)
│   ├── parameters.jl       # Physical parameters
│   ├── Helpers.jl          # Helper functions & coordinate transforms
│   ├── GellMann.jl         # SU(N) generators & structure constants
│   ├── Wavefunctions.jl    # Light-cone baryon wavefunctions
│   ├── EMFormFactors.jl    # Electromagnetic form factors (F1, F2)
│   ├── Odderon.jl          # Cubic color correlator, odderon distribution, gluon Sivers
│   └── Pomeron.jl          # Placeholder for pomeron exchange physics
├── scripts/
│   └── writers.jl          # Parallel batch computation scripts
└── notebooks/
    └── plots.ipynb         # Plot generation using Python's matplotlib
```

