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
    norm = 21693.23305361701,   # Baryon wavefunction norm from normalize_wavefunction()
    mq = 0.26,                  # Constituent quark mass
    β = 0.55,                   # Baryon wavefunction width
    Nc = 3,                     # Number of colors
    αs = 0.25,                  # g^2 / (4π)
    mN = 0.93827                # Nucleon mass in GeV
    ) 

# ======================
end # end module