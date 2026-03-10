"""
    NucleonLCCQM

Julia package for computing observables in a light-cone constituent quark model (LC-CQM).

Submodules:
- `Wavefunctions`: baryon wavefunctions (spin, momentum-space, normalization)
- `EMFormFactors`: electromagnetic Dirac (F1) and Pauli (F2) form factors
- `Odderon`: cubic color correlator, odderon distribution, gluon Sivers function
- `Pomeron`: placeholder for future pomeron exchange physics
- `Helpers`: coordinate transforms, Cuba solver map, utility functions
- `GellMann`: SU(N) generators and structure constants

## Quick start
```julia
using NucleonLCCQM

# Compute F1 form factor at Δ = [0.1, 0] GeV
f1, err, prob, neval, fail, nregions = f1_form_factor([0.1, 0.0])

# Compute gluon Sivers at k = [0.5, 0] GeV
res, err, prob, neval, fail, nregions = gluon_sivers([0.5, 0.0]; μ=0.0, solver=:vegas)
```

## Configuration
Model parameters (masses, wavefunction type, normalizations) are defined in `src/parameters.jl`.
Adjust WF_TYPE, MQ, BETA, NORM, etc. as needed, then use result of normalize_wavefunction() to ensure proper normalization.
"""
module NucleonLCCQM

# ======================
# Shared parameters (defines MQ, BETA, NORM, NC, etc. in this module's scope)
# ======================
include("parameters.jl")

# ======================
# Utility modules
# ======================
include("Helpers.jl")
include("GellMann.jl")

# ======================
# Physics modules
# ======================
include("Wavefunctions.jl")
include("EMFormFactors.jl")
include("Odderon.jl")
include("Pomeron.jl")

# ======================
# Re-export all public symbols from submodules
# ======================
using .Wavefunctions
using .EMFormFactors
using .Odderon
using .Pomeron

# ======================
# Exports
# ======================

# Submodule names for qualified access (e.g. NucleonLCCQM.Odderon.gluon_sivers)
export Helpers, GellMann, Wavefunctions, EMFormFactors, Odderon, Pomeron

# Re-exported from Wavefunctions
export spin_wavefunction,
       momentum_space_wavefunction,
       baryon_wavefunction,
       compute_wavefunction,
       spin_sum,
       normalize_wavefunction

# Re-exported from EMFormFactors
export f_form_factor,
       f1_form_factor,
       f2_form_factor

# Re-exported from Odderon
export cubic_color_correlator,
       integrate_cubic_color_correlator,
       ft_cubic_color_correlator,
       odderon_distribution,
       odderon_distribution_r,
       gluon_sivers

# Re-exported from Pomeron
export quadratic_color_correlator

# ======================
end # module NucleonLCCQM