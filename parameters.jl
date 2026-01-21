module Parameters

# ======================
# Exports
# ======================

export params

# ======================
# Parameter definitions
# ======================

# Add as needed
const params = (
    # ======================
    # Original parameters
    # ======================
    # wf_type = :exp,            # Wavefunction type
    # mq = 0.26,                 # Constituent quark mass
    # β = 0.55,                  # Baryon wavefunction width
    # norm = 21693.23305361701,  # Baryon wavefunction norm from normalize_wavefunction()
    # ======================
    # Power-like wfs.
    # ======================
    wf_type = :pow,            # Wavefunction type
    p = 3.5,                   # Power parameter for power-like wavefunction
    mq = 0.28,                 # Constituent quark mass
    β = 0.88,                  # Baryon wavefunction width
    norm = 87088.44473120169,  # Baryon wavefunction norm from normalize_wavefunction()
    # ======================
    # Exp. wfs.
    # ======================
    # wf_type = :exp,            # Wavefunction type
    # mq = 0.24,                 # Constituent quark mass
    # β = 0.7,                   # Baryon wavefunction width
    # norm = 7389.461074380284,  # Baryon wavefunction norm from normalize_wavefunction()
    # ======================
    # Common parameters
    # ======================
    Nc = 3,                    # Number of colors
    αs = 0.25,                 # g^2 / (4π), So far not used
    mN = 0.93827               # Nucleon mass in GeV
    ) 

# ======================
end # end module