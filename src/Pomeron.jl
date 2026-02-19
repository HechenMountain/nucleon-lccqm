"""
    Pomeron

Placeholder module for future pomeron exchange physics
in the light-cone constituent quark model.
"""
module Pomeron

# ======================
# Imports
# ======================

using SpecialFunctions: besselj1

# Import parameters from parent module
import ..NORM, ..NC, ..ALPHA_S, ..M_N

# Import sibling modules
import ..Helpers as hp
import ..Wavefunctions as wfs

# ======================
# Exports
# ======================

# (none yet)

# ======================
# Distributions
# ======================
"""
    quadratic_color_correlator(s01::Integer,s02::Integer,
                           q1::AbstractVector{<:Real}, q2::AbstractVector{<:Real}, q3::AbstractVector{<:Real},
                           x1::Real, x2::Real, x3::Real,
                           k1::AbstractVector{<:Real}, k2::AbstractVector{<:Real}, k3::AbstractVector{<:Real})

Compute the unintegrated quadratic color correlator by summing one-, two-, and three-body contributions.

Arguments
- `s01, s02`: Spins of the ingoing/outgoing protons (each must be either +1 or -1)
- `q1, q2, q3`: Eikonal momenta (2D cartesian vectors)
- `x1, x2, x3`: Parton x values satisfying x1 + x2 + x3 = 1
- `k1, k2, k3`: Transverse momenta (2D cartesian vectors)

Returns
- `ccc::ComplexF64`: Value of the quadratic color correlator for the given spin configuration and kinematics

Notes
- This is G_2ΛΛ' stripped of the integrals and factor of 1/(4π)^2/(2π)^2 in the draft
- Momenta must be in cartesian coordinates
"""
function quadratic_color_correlator(s01::Integer,s02::Integer,
                                q1::AbstractVector{<:Real}, q2::AbstractVector{<:Real},
                                x1::Real, x2::Real, x3::Real,
                                k1::AbstractVector{<:Real}, k2::AbstractVector{<:Real}, k3::AbstractVector{<:Real})
    function one_body_kin(i, j12, q1, q2)
        # Momentum inflow [q1 + q2] at j12
        # i is k_prime (quark) index, j12 quark line with 
        # momentum inflow q1 + q2 from gluon.
        delta_kiprime =  hp.δ(i, j12) * (q1 + q2)
        return delta_kiprime
    end
    function two_body_kin(i, j1, j2, q1, q2)
        # Momentum inflow [q1,j1] [q2,j2] at j1 and j2
        # i is k_prime (quark) index, j1, j2 quark line with
        # momentum inflow q1 and q2 from gluon.
        delta_kiprime = hp.δ(i, j1) * q1 + hp.δ(i, j2) * q2
        return delta_kiprime
    end
    # Quadratic color correlator without Jacobian
    # Precompute incoming baryon wavefunction
    wf1 = wfs.compute_wavefunction(s01, x1, x2, x3, k1, k2, k3)

    # Constant parts
    Δ = - (q1 + q2)
    k1prime0, k2prime0, k3prime0 = k1 - x1 * Δ, k2 - x2 * Δ, k3 - x3 * Δ

    qcc = complex(0,0)
    # One-body
    for j12 in 1:3
        k1prime = k1prime0 - one_body_kin(1, j12, q1, q2)
        k2prime = k2prime0 - one_body_kin(2, j12, q1, q2)
        k3prime = k3prime0 - one_body_kin(3, j12, q1, q2)
        wf2 = wfs.compute_wavefunction(s02, x1, x2, x3, k1prime, k2prime, k3prime)
        # Perform spin sum
        qcc += wfs.spin_sum(wf1, wf2)
    end
    # Two-body
    for j1 in 1:3, j2 in 1:3
        if j1 == j2
            continue
        end
        k1prime = k1prime0 - two_body_kin(1, j1, j2, q1, q2)
        k2prime = k2prime0 - two_body_kin(2, j1, j2, q1, q2)
        k3prime = k3prime0 - two_body_kin(3, j1, j2, q1, q2)
        wf2 = wfs.compute_wavefunction(s02, x1, x2, x3, k1prime, k2prime, k3prime)
        # Perform spin sum
        qcc -= .5 * wfs.spin_sum(wf1, wf2)
    end
    return qcc
end

# ======================
end # module Pomeron
