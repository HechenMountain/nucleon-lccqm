# Sivers.jl

Julia package for computing the gluon Sivers function and related form factors in a light-cone constituent quark model.

## About
We parametrize the light-cone baryon wavefunction in a truncated three-quark Fock sector, compute the electromagnetic Dirac/Pauli form factors, build the cubic color correlator, and integrate it to obtain the gluon Sivers function at non-vanishing transverse momentum transfer.

## Installation

From the Julia REPL:
```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/Sivers.jl")
```

Or for development:
```julia
Pkg.develop(path="/path/to/Sivers.jl")
```

## Features
- Symbol-based configuration for solvers (`:cuhre`, `:vegas`, `:suave`, `:divonne`)
- Linear/log spacing controls for writers and sampling utilities (`spacing=:lin` or `:log`)
- Multiple wavefunction parametrizations selectable in `src/Parameters.jl` (`WF_TYPE = :pow` or `:exp`)
- Parallel CSV writers for form factors, odderon distributions, and the Sivers function

## Quick start

```julia
using Sivers

# Example: gluon Sivers at k=0.5 GeV with vegas
res, err, prob, neval, fail, nregions = gluon_sivers(0.5; μ=0.0, solver=:vegas)

# Form factors
f1, err_f1, prob, neval, fail, nregions = f1_form_factor([0.1, 0.0])
f2, err_f2, prob, neval, fail, nregions = f2_form_factor([0.1, 0.0])

# Normalize wavefunction
norm, err, prob, neval, fail, nregions = normalize_wavefunction()
```

## Batch runs
Use the writer script with parallel workers:
```sh
julia -p 4 scripts/writers.jl
```
Uncomment or edit the desired `write_*` calls near the end of `scripts/writers.jl`. All solver arguments are Symbols (e.g., `solver=:vegas`), and spacing is controlled with `spacing=:lin` or `:log`.

## Notebooks
See `notebooks/sivers.ipynb` for interactive examples.

## Configuration
Adjust model parameters in `src/Parameters.jl`:
- `WF_TYPE`: wavefunction type (`:pow` or `:exp`)
- `MQ`: constituent quark mass
- `BETA`: baryon wavefunction width parameter
- `NORM`: wavefunction normalization (from `normalize_wavefunction()`)
- `NC`: number of colors
- `ALPHA_S`: strong coupling constant
- `M_N`: nucleon mass

## Project Structure
```
Sivers.jl/
├── Project.toml           # Package metadata & dependencies
├── src/
│   ├── Sivers.jl          # Main module (entry point)
│   ├── core.jl            # Core implementation
│   ├── Parameters.jl      # Physical parameters
│   ├── Helpers.jl         # Helper functions & coordinate transforms
│   ├── LightConeCQM.jl    # Light-cone wavefunction module
│   └── GellMann.jl        # SU(N) generators
├── scripts/
│   └── writers.jl         # Parallel batch computation scripts
└── notebooks/
    └── sivers.ipynb       # Interactive examples
```

