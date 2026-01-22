# ======================
# Parameter definitions
# ======================

# Add as needed

# ======================
# Original parameters
# ======================
# const WF_TYPE = :exp          # Wavefunction type
# const MQ = 0.26               # Constituent quark mass
# const BETA = 0.55             # Baryon wavefunction width
# const NORM = 21693.23305361701 # Baryon wavefunction norm from normalize_wavefunction()

# ======================
# Power-like wfs (active)
# ======================
const WF_TYPE = :pow           # Wavefunction type
const P = 3.5                  # Power parameter for power-like wavefunction
const MQ = 0.28                # Constituent quark mass
const BETA = 0.88              # Baryon wavefunction width
const NORM = 87088.44473120169 # Baryon wavefunction norm from normalize_wavefunction()

# ======================
# Exp. wfs.
# ======================
# const WF_TYPE = :exp          # Wavefunction type
# const MQ = 0.24               # Constituent quark mass
# const BETA = 0.7              # Baryon wavefunction width
# const NORM = 7389.461074380284 # Baryon wavefunction norm from normalize_wavefunction()

# ======================
# Common parameters
# ======================
const NC = 3                   # Number of colors
const ALPHA_S = 0.25           # g^2 / (4π), So far not used
const M_N = 0.93827            # Nucleon mass in GeV