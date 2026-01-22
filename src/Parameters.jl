"""
    Parameters

Physical parameters for the Sivers function calculation.

## Wavefunction types
- `:exp`: Exponential wavefunction
- `:pow`: Power-law wavefunction

## Parameter sets
Uncomment the desired parameter set below. The normalization constant `NORM`
should be computed using `normalize_wavefunction()` for each parameter set.
"""
module Parameters

# Export all parameters
export WF_TYPE, P, MQ, BETA, NORM, NC, ALPHA_S, M_N

# ======================
# Original parameters
# ======================
# const WF_TYPE = :exp          # Wavefunction type
# const MQ = 0.26               # Constituent quark mass (GeV)
# const BETA = 0.55             # Baryon wavefunction width (GeV)
# const NORM = 21693.23305361701 # Baryon wavefunction norm from normalize_wavefunction()

# ======================
# Power-like wfs
# ======================
const WF_TYPE = :pow           # Wavefunction type
const P = 3.5                  # Power parameter for power-like wavefunction
const MQ = 0.28                # Constituent quark mass (GeV)
const BETA = 0.88              # Baryon wavefunction width (GeV)
const NORM = 87088.44473120169 # Baryon wavefunction norm from normalize_wavefunction()

# ======================
# Exp. wfs.
# ======================
# const WF_TYPE = :exp          # Wavefunction type
# const MQ = 0.24               # Constituent quark mass (GeV)
# const BETA = 0.7              # Baryon wavefunction width (GeV)
# const NORM = 7389.461074380284 # Baryon wavefunction norm from normalize_wavefunction()

# ======================
# Common parameters
# ======================
const NC = 3                   # Number of colors
const ALPHA_S = 0.25           # g^2 / (4π), Strong coupling constant
const M_N = 0.93827            # Nucleon mass (GeV)

# ======================
end # module Parameters