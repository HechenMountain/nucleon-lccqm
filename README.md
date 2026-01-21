# sivers.jl

Julia code to compute the gluon Sivers function and related form factors in a light-cone constituent quark model.

## About
We parametrize the light-cone baryon wavefunction in a truncated three-quark Fock sector, compute the electromagnetic Dirac/Pauli form factors, build the cubic color correlator, and integrate it to obtain the gluon Sivers function at non-vanishing transverse momentum transfer.

## Features
- Symbol-based configuration for solvers (`:cuhre`, `:vegas`, `:suave`, `:divonne`) via the shared `Helpers.SOLVERS` map
- Linear/log spacing controls for writers and sampling utilities (`spacing=:lin` or `:log`)
- Multiple wavefunction parametrizations selectable in `parameters.jl` (`wf_type = :pow` or `:exp`)
- Parallel CSV writers for form factors, odderon distributions, and the Sivers function

## Quick start (Julia REPL)
```julia
include("core.jl")
using .Sivers

# Example: gluon Sivers at k=0.5 GeV with vegas
res, err, prob, neval, fail, nregions = gluon_sivers(0.5; μ=0.0, solver=:vegas)
```

## Batch runs
Use the writer script with parallel workers:
```sh
julia -p 4 writers.jl
```
Uncomment or edit the desired `write_*` calls near the end of `writers.jl`. All solver arguments are Symbols (e.g., `solver=:vegas`), and spacing is controlled with `spacing=:lin` or `:log`.

## Notebooks
See `sivers.ipynb` for interactive examples. It now uses symbol-based solver selection consistent with the Julia API.

## Configuration
Adjust model parameters in `parameters.jl` (mass scales, wavefunction type, normalizations). The shared solver map lives in `helpers.jl` as `SOLVERS`.

