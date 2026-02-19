"""
    Odderon

Module for computing the odderon distribution and gluon Sivers function
in the light-cone constituent quark model.

## Main exports
- `cubic_color_correlator`: unintegrated three-gluon correlator
- `integrate_cubic_color_correlator`: integrated cubic color correlator
- `ft_cubic_color_correlator`: Fourier-transformed cubic color correlator
- `odderon_distribution`: odderon distribution in momentum space
- `odderon_distribution_r`: odderon distribution in position space
- `gluon_sivers`: gluon Sivers function
"""
module Odderon

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

export cubic_color_correlator,
       integrate_cubic_color_correlator,
       ft_cubic_color_correlator,
       odderon_distribution,
       odderon_distribution_r,
       gluon_sivers

# ======================
# Constants
# ======================

# SU(NC) color factor for the odderon
const dabc2 = (NC^2 - 4) * (NC^2 - 1) / NC

# ======================
# Distributions
# ======================

"""
    cubic_color_correlator(s01::Integer,s02::Integer,
                           q1::AbstractVector{<:Real}, q2::AbstractVector{<:Real}, q3::AbstractVector{<:Real},
                           x1::Real, x2::Real, x3::Real,
                           k1::AbstractVector{<:Real}, k2::AbstractVector{<:Real}, k3::AbstractVector{<:Real})

Compute the unintegrated cubic color correlator by summing one-, two-, and three-body contributions.

Arguments
- `s01, s02`: Spins of the ingoing/outgoing protons (each must be either +1 or -1)
- `q1, q2, q3`: Eikonal momenta (2D cartesian vectors)
- `x1, x2, x3`: Parton x values satisfying x1 + x2 + x3 = 1
- `k1, k2, k3`: Transverse momenta (2D cartesian vectors)

Returns
- `ccc::ComplexF64`: Value of the cubic color correlator for the given spin configuration and kinematics

Notes
- This is G_3ΛΛ' stripped of the integrals and factor of 1/(4π)^2/(2π)^2 in the draft
- Momenta must be in cartesian coordinates
"""
function cubic_color_correlator(s01::Integer,s02::Integer,
                                q1::AbstractVector{<:Real}, q2::AbstractVector{<:Real}, q3::AbstractVector{<:Real},
                                x1::Real, x2::Real, x3::Real,
                                k1::AbstractVector{<:Real}, k2::AbstractVector{<:Real}, k3::AbstractVector{<:Real})
    function one_body_kin(i, j123, q1, q2, q3)
        # Momentum inflow [q1 + q2 + q3] at j123
        # i is k_prime (quark) index, j123 quark line with 
        # momentum inflow q1 + q2 + q3 from gluon.
        delta_kiprime =  hp.δ(i, j123) * (q1 + q2 + q3)
        return delta_kiprime
    end
    function two_body_kin(i, j12, j3, l, q1, q2, q3)
        # Momentum inflow [q1 + q2,j12] [q3,j3] at j12 and j3
        # i is k_prime (quark) index, j12, j3 quark line with
        # momentum inflow q1 + q2 and q3 from gluon.
        # Addtional terms from permutations, so in total 3 contributions
        # which we distinguish by l
        if l == 1 # [q2 + q3,j12] [q1,j3]
            delta_kiprime =  hp.δ(i, j12) * (q2 + q3) + hp.δ(i, j3) * q1
        elseif l == 2 # [q1 + q3,j12] [q2,j3]
            delta_kiprime =  hp.δ(i, j12) * (q1 + q3) + hp.δ(i, j3) * q2
        elseif l == 3 # [q1 + q2,j12] [q3,j3]
            delta_kiprime =  hp.δ(i, j12) * (q1 + q2) + hp.δ(i, j3) * q3
        end
        return delta_kiprime
    end
    function three_body_kin(i, j1, j2, j3, q1, q2, q3)
        # Momentum inflow [q1,j1] [q2,j2] [q3,j3]
        # i is k_prime (quark) index, j1, j2, j3 are gluons with momenta
        # q1, q2 and q3, respectively, attached to quark lines.
        delta_kiprime = hp.δ(i, j1) * q1 + hp.δ(i, j2) * q2 + hp.δ(i, j3) * q3
        return delta_kiprime
    end
    # Cubic color correlator without Jacobian
    # Precompute incoming baryon wavefunction
    wf1 = wfs.compute_wavefunction(s01, x1, x2, x3, k1, k2, k3)

    # Constant parts
    Δ = - (q1 + q2 + q3)
    k1prime0, k2prime0, k3prime0 = k1 - x1 * Δ, k2 - x2 * Δ, k3 - x3 * Δ
    # Sum over one-body, two-body and three-body kinematics
    ccc = complex(0,0)
    # One-body
    for j123 in 1:3
        k1prime = k1prime0 - one_body_kin(1, j123, q1, q2, q3)
        k2prime = k2prime0 - one_body_kin(2, j123, q1, q2, q3)
        k3prime = k3prime0 - one_body_kin(3, j123, q1, q2, q3)
        wf2 = wfs.compute_wavefunction(s02, x1, x2, x3, k1prime, k2prime, k3prime)
        # Perform spin sum
        ccc += wfs.spin_sum(wf1, wf2)
    end
    # Two-body
    # Addtional terms from permutations, so we have an extra sum over l
    for l in 1:3, j12 in 1:3, j3 in 1:3
        if j12 == j3
            continue
        end
        k1prime = k1prime0 - two_body_kin(1, j12, j3, l, q1, q2, q3)
        k2prime = k2prime0 - two_body_kin(2, j12, j3, l, q1, q2, q3)
        k3prime = k3prime0 - two_body_kin(3, j12, j3, l, q1, q2, q3)
        wf2 = wfs.compute_wavefunction(s02, x1, x2, x3, k1prime, k2prime, k3prime)
        # Perform spin sum
        ccc -= 0.5 * wfs.spin_sum(wf1, wf2)
    end
    # Three-body
    for j1 in 1:3, j2 in 1:3, j3 in 1:3
        if j1 == j2 || j1 == j3 || j2 == j3
            continue
        end
        k1prime = k1prime0 - three_body_kin(1, j1, j2, j3, q1, q2, q3)
        k2prime = k2prime0 - three_body_kin(2, j1, j2, j3, q1, q2, q3)
        k3prime = k3prime0 - three_body_kin(3, j1, j2, j3, q1, q2, q3)
        wf2 = wfs.compute_wavefunction(s02, x1, x2, x3, k1prime, k2prime, k3prime)
        # Perform spin sum
        ccc += wfs.spin_sum(wf1, wf2)
    end
    return ccc
end

"""
    integrate_cubic_color_correlator(s01::Integer,s02::Integer,
                                     q1::AbstractVector{<:Real}, q2::AbstractVector{<:Real}, q3::AbstractVector{<:Real};
                                     solver::Symbol=:vegas)

Compute the integrated cubic color correlator.

Arguments
- `s01, s02`: Spins of the ingoing/outgoing protons (each must be either +1 or -1)
- `q1, q2, q3`: Eikonal momenta (2D cartesian vectors)
- `solver`: Integration strategy. Options: :cuhre, :vegas, :divonne, :suave. Default is :vegas

Returns
- ToDo

Notes
- This is G_3ΛΛ' including the integrals to run some checks.
- Momenta must be in cartesian coordinates
"""
function integrate_cubic_color_correlator(s01::Integer,s02::Integer,
                                          q1::AbstractVector{<:Real}, q2::AbstractVector{<:Real}, q3::AbstractVector{<:Real};
                                          solver::Symbol=:vegas)
    sol = get(hp.SOLVERS, solver) do
        throw(ArgumentError("solver must be one of: :cuhre, :vegas, :divonne, :suave"))
    end

    function integrand(x,f)
        # Regulate endpoint singularities
        x = hp.regulate_cuba(x)
        # Transform [0,1]^6 cuba samples to physical variables
        # Parton-x
        (x1, x2, x3), d2x = hp.cuba_to_parton_x(x[1:2])
        # Momenta
        r1, ϕ1, d2k1 = hp.cuba_to_polar(x[3:4])    # k1
        r2, ϕ2, d2k2 = hp.cuba_to_polar(x[5:6])    # k2
    
        # Reconstruct cartesian momenta from polar coordinates
        k1 = hp.polar_to_cartesian(hp.vec2(r1, ϕ1))
        k2 = hp.polar_to_cartesian(hp.vec2(r2, ϕ2))
        k3 = - (k1 + k2)  # Enforce transverse momentum conservation
        d4k = d2k1 * d2k2

        ccc = cubic_color_correlator(s01, s02, q1, q2, q3, x1, x2, x3, k1, k2, k3)

        # Jacobian
        d6x = d2x * d4k # 2 + 4 = 6d integral
        ccc *=  d6x

        f[1] = real(ccc)
        f[2] = imag(ccc)
    end
    integral, err, prob, neval, fail, nregions = sol(integrand, 6, 2; rtol=9e-3, maxevals=10_000_000)
    # Prefactors 
    wf_norm = NORM
    prf = wf_norm^2
    # π factors from integration:
    # Deltas have (2π)^2 * 4π in front. 
    # Every integration over k gives 1 / (2π)^2 -> 1 / (2π)^(2 * 3)
    # Every integration over x gives 1 / (4π)
    prf /= (4π)^2 * (2π)^4
    # We return the result up to a factor of k^2
    integral .*= prf
    err .*= abs(prf)
    return integral, err, prob, neval, fail, nregions 
end

"""
    ft_cubic_color_correlator(s01::Integer,s02::Integer,
                              rabs::Real, solver::Symbol=:vegas)

Compute the Fourier transform of the integrated cubic color correlator.

Arguments
- `s01, s02`: Spins of the ingoing/outgoing protons (each must be either +1 or -1)
- `rabs`: Absolute value of  transverse coordinate vector
- `solver`: Integration strategy. Options: :cuhre, :vegas, :divonne, :suave. Default is :vegas

Returns
- ToDo

Notes
- We are Fourier transforming and integrating the cubic correlator in one go because
  of the long tail in kT
- Uses the Hankel transform to reduce dimensionality
- Momenta must be in cartesian coordinates
"""
function ft_cubic_color_correlator(s01::Integer,s02::Integer,
                                   rabs::Real; solver::Symbol=:vegas)
    sol = get(hp.SOLVERS, solver) do
        throw(ArgumentError("solver must be one of: :cuhre, :vegas, :divonne, :suave"))
    end

    function integrand(x,f)
        # Regulate endpoint singularities
        x = hp.regulate_cuba(x)
        # Transform [0,1]^8 cuba samples to physical variables
        # Parton-x
        (x1, x2, x3), d2x = hp.cuba_to_parton_x(x[1:2])
        # Momenta
        r1, ϕ1, d2k1 = hp.cuba_to_polar(x[3:4])      # k1
        r2, ϕ2, d2k2 = hp.cuba_to_polar(x[5:6])      # k2
        r3, dΔ = x[7] / (1 - x[7]), 1 / (1 - x[7])^2 # Δ
    
        # Reconstruct cartesian momenta from polar coordinates
        k1 = hp.polar_to_cartesian(hp.vec2(r1, ϕ1))
        k2 = hp.polar_to_cartesian(hp.vec2(r2, ϕ2))
        k3 = - (k1 + k2)    # Enforce transverse momentum conservation
        d4k = d2k1 * d2k2

        Δ = hp.vec2(r3, 0.0)
        # Assume q_12 = q_23 = 0 
        q12, q23 = hp.vec2(0.0, 0.0), hp.vec2(0.0, 0.0)
        q1, q2, q3 = (2 * q12 + q23 - Δ) / 3, (- q12 + q23 - Δ) / 3, - (q12 + 2 * q23 + Δ) / 3
        # Jacobian
        d7x = d2x * d4k * dΔ   # 2 + 4 + 1 = 7d integral

        # Cubic color correlator
        ccc = cubic_color_correlator(s01, s02, q1, q2, q3, x1, x2, x3, k1, k2, k3)

        # Hankel factor
        Δabs = sqrt(hp.sqnorm2(Δ))
        h_factor = im * Δabs * besselj1(Δabs * rabs)

        # Full result
        res = ccc * h_factor * d7x

        f[1] = real(res)
        f[2] = imag(res)
    end
    integral, err, prob, neval, fail, nregions = sol(integrand, 7, 2; rtol=9e-3, maxevals=100_000_000)
    # Prefactors 
    wf_norm = NORM
    prf = wf_norm^2
    # π factors from integration:
    # Deltas have (2π)^2 * 4π in front. 
    # Every integration over k gives 1 / (2π)^2 -> 1 / (2π)^(2 * 3)
    # Every integration over x gives 1 / (4π)
    prf /= (4π)^2 * (2π)^5
    integral .*= prf
    err .*= abs(prf)
    return integral, err, prob, neval, fail, nregions 
end

"""
    odderon_distribution(s01::Integer,s02::Integer,
                         Δ::AbstractVector{<:Real}, k::AbstractVector{<:Real};
                         μ::Real=0.00, solver::Symbol=:vegas)

Compute the Odderon distribution O * k^2 for momentum transfer k and Δ.

Arguments
- `s01, s02`: Spin of ingoing/outgoing proton (each must be either +1 or -1)
- `Δ`: 2D momentum transfer vector in cartesian coordinates
- `k`: 2D transverse momentum transfer vector in cartesian coordinates
- `μ`: Regulator for integrand (default: 0.00)
- `solver`: Integration strategy. Options: :cuhre, :vegas, :divonne, :suave. Default is :vegas

Returns
- `integral::Vector{Float64}`: Array [re, im] containing real and imaginary parts of O(k,Δ) * k^2
- `err::Vector{Float64}`: Array [err_re, err_im] containing error estimates for real and imaginary parts
- `prob::Vector{Float64}`: Array [prob_re, prob_im] containing probability estimates for each component
- `neval::Int64`: Number of integrand evaluations
- `fail::Int32`: Integration failure flag (0 = success)
- `nregions::Int32`: Number of subregions used in the integration

Notes
- Currently only valid for Δ = [0,0], corresponds to OΛΛ'(k,Δ) in the draft
- Factor 1/k^2 is omitted here and added later in the Sivers function definition
- Momenta must be in cartesian coordinates
- `NORM` is set in `parameters.jl`, obtainable via `normalize_wavefunction()`
- Result is generally complex; for k_y = 0 it is real
- We enforce the symmetry in the k integration to get a simpler integrand
- Factor g^6 = 1
"""
function odderon_distribution(s01::Integer,s02::Integer,
                              Δ::AbstractVector{<:Real}, k::AbstractVector{<:Real};
                              μ::Real=0.00,solver::Symbol=:vegas)
    if !all(iszero, Δ)
        throw(ArgumentError("Implementation currently only for vanishing Δ."))
    end
    
    sol = get(hp.SOLVERS, solver) do
        throw(ArgumentError("solver must be one of: :cuhre, :vegas, :divonne, :suave"))
    end

    function integrand(x,f)
        # Regulate endpoint singularities
        x = hp.regulate_cuba(x)
        # Transform [0,1]^8 cuba samples to physical variables
        # Parton-x
        (x1, x2, x3), d2x = hp.cuba_to_parton_x(x[1:2])
        # Momenta
        r1, ϕ1, d2k1 = hp.cuba_to_polar(x[3:4])    # k1
        r2, ϕ2, d2k2 = hp.cuba_to_polar(x[5:6])    # k2
        r3, ϕ3, d2q2 = hp.cuba_to_polar(x[7:8])    # q2
        
        # Reconstruct cartesian momenta from polar coordinates
        k1 = hp.polar_to_cartesian(hp.vec2(r1, ϕ1))
        k2 = hp.polar_to_cartesian(hp.vec2(r2, ϕ2))
        k3 = - (k1 + k2)  # Enforce transverse momentum conservation
        d4k = d2k1 * d2k2

        q2 = hp.polar_to_cartesian(hp.vec2(r3, ϕ3))
        # Jacobian
        d8x = d2x * d4k * d2q2 # 2 + 4 + 2 = 8d integral
        μ2 = μ^2 # Regulator squared
        
        total = 0
        # Simplified integrand for spin-flip
        if s01 != s02
            for s in (+1,-1)
                # Flip momenta to project out Sivers function
                q1, q2, q3 = s * k, s * q2, - s * (k + q2)
                ccc = cubic_color_correlator(s01, s02, q1, q2, q3, x1, x2, x3, s*k1, s*k2, s*k3)
                total += s * ccc
            end
            # Regenerate initial q2
            q2 = hp.polar_to_cartesian(hp.vec2(r3, ϕ3))
            q3 = k + q2
            q22, q32 = hp.sqnorm2(q2), hp.sqnorm2(q3)
            # Add regulator
            q22 += μ2
            q32 += μ2
            # Same denominator
            # for both terms once momenta
            # have been flipped
            den = q22 * q32 
            total /= den
        else
            # This part can probably also be simplified
            q22 = hp.sqnorm2(q2)
            # Add regulator
            q22 += μ2
            for s in (+1,-1)
                q1, q3 = s * k, - (s * k + q2)
                ccc = cubic_color_correlator(s01, s02, q1, q2, q3, x1, x2, x3, k1, k2, k3)
                q32 = hp.sqnorm2(q3)
                # Add regulator
                q32 += μ2
                total += s * ccc / q32
            end
            total /= q22
        end
        total *=  d8x

        f[1] = real(total)
        f[2] = imag(total)
    end
    integral, err, prob, neval, fail, nregions = sol(integrand, 8, 2; rtol=9e-3, maxevals=10_000_000)
    # Prefactors 
    wf_norm = NORM
    # Factor of g^6 = 1, for simplicity, treated later on
    prf = - dabc2 / NC / 32 * wf_norm^2
    # π factors from integration:
    # Deltas have (2π)^2 * 4π in front. 
    # Every integration over k gives 1 / (2π)^2 -> 1 / (2π)^(2 * 3)
    # Every integration over x gives 1 / (4π)
    # Final integration over q gives 1 / (2π)^2
    prf /= (4π)^2 * (2π)^6
    # We return the result up to a factor of k^2
    integral .*= prf
    err .*= abs(prf)
    return integral, err, prob, neval, fail, nregions 
end

"""
    odderon_distribution_r(s01::Integer,s02::Integer,
                         Δ::AbstractVector{<:Real}, r::AbstractVector{<:Real};
                         μ::Real=0.00, solver::Symbol=:vegas)

Compute the Odderon distribution O in position space r and Δ.

Arguments
- `s01, s02`: Spin of ingoing/outgoing proton (each must be either +1 or -1)
- `Δ`: 2D momentum transfer vector in cartesian coordinates
- `r`: 2D position vector in cartesian coordinates
- `μ`: Regulator for integrand (default: 0.00)
- `solver`: Integration strategy. Options: :cuhre, :vegas, :divonne, :suave. Default is :vegas

Returns
- `integral::Vector{Float64}`: Array [re, im] containing real and imaginary parts of O(r,Δ)
- `err::Vector{Float64}`: Array [err_re, err_im] containing error estimates for real and imaginary parts
- `prob::Vector{Float64}`: Array [prob_re, prob_im] containing probability estimates for each component
- `neval::Int64`: Number of integrand evaluations
- `fail::Int32`: Integration failure flag (0 = success)
- `nregions::Int32`: Number of subregions used in the integration

Notes
- vectors must be in cartesian coordinates
- `NORM` is set in `parameters.jl`, obtainable via `normalize_wavefunction()`
- Result is generally complex
- Factor ig^6 = 1
"""
function odderon_distribution_r(s01::Integer,s02::Integer,
                              Δ::AbstractVector{<:Real}, r::AbstractVector{<:Real};
                              μ::Real=0.00,solver::Symbol=:vegas)
    sol = get(hp.SOLVERS, solver) do
        throw(ArgumentError("solver must be one of: :cuhre, :vegas, :divonne, :suave"))
    end

    function integrand(x,f)
        # Regulate endpoint singularities
        x = hp.regulate_cuba(x)
        # Transform [0,1]^8 cuba samples to physical variables
        # Parton-x
        (x1, x2, x3), d2x = hp.cuba_to_parton_x(x[1:2])
        # Momenta
        r1, ϕ1, d2k1 = hp.cuba_to_polar(x[3:4])    # k1
        r2, ϕ2, d2k2 = hp.cuba_to_polar(x[5:6])    # k2
        r3, ϕ3, d2q1 = hp.cuba_to_polar(x[7:8])    # q1
        r4, ϕ4, d2q2 = hp.cuba_to_polar(x[9:10])   # q2
        
        # Reconstruct cartesian momenta from polar coordinates
        k1 = hp.polar_to_cartesian(hp.vec2(r1, ϕ1))
        k2 = hp.polar_to_cartesian(hp.vec2(r2, ϕ2))
        k3 = - (k1 + k2)  # Enforce transverse momentum conservation
        d4k = d2k1 * d2k2

        q1 = hp.polar_to_cartesian(hp.vec2(r3, ϕ3))
        q2 = hp.polar_to_cartesian(hp.vec2(r4, ϕ4))
        q3 = Δ - (q1 + q2)
        d4q = d2q1 * d2q2

        μ2 = μ^2 # Regulator squared
        q12, q22, q32 = hp.sqnorm2(q1) + μ2, hp.sqnorm2(q2) + μ2, hp.sqnorm2(q3) + μ2
        # Jacobian
        d10x = d2x * d4k * d4q # 2 + 4 + 4 = 10d integral
        
        # Simplified integrand for spin-flip
        total = cubic_color_correlator(s01, s02, q1, q2, q3, x1, x2, x3, k1, k2, k3)
        den = q12 * q22 * q32
        # Simplified integrand in forward limit
        if all(iszero, Δ)
            trig1, trig2, trig3 = sin(.5 * q1'r), sin(.5 * q2'r), sin(.5 * q3'r)
            trig = 4 / 3 * trig1 * trig2 * trig3
        else
            trig1, trig2, trig3, trig4 = sin(.5 * Δ'r), sin(.5 * (2q1 - Δ)'r), sin(.5 * (2q2 - Δ)'r), sin(.5 * (2q3 - Δ)'r)
            trig = trig1 + trig2 + trig3 + trig4
            trig *= 1 / 3
        end
        total *= trig * d10x / den

        f[1] = real(total)
        f[2] = imag(total)
    end
    integral, err, prob, neval, fail, nregions = sol(integrand, 10, 2; rtol=9e-3, maxevals=70_000_000)
    # Prefactors 
    wf_norm = NORM
    # Factor of ig^6 = 1, for simplicity, treated later on
    prf = - dabc2 / NC / 16 * wf_norm^2
    # π factors from integration:
    # Deltas have (2π)^2 * 4π in front. 
    # Every integration over k gives 1 / (2π)^2 -> 1 / (2π)^(2 * 3)
    # Every integration over x gives 1 / (4π)
    # Final integration over q gives 1 / (2π)^2
    prf /= (4π)^2 * (2π)^8

    integral .*= prf
    err .*= abs(prf)
    return integral, err, prob, neval, fail, nregions 
end

"""
    gluon_sivers(k::AbstractVector{<:Real}; μ::Real=0.00, solver::Symbol=:vegas)

Compute the gluon Sivers function for momentum transfer k.

Arguments
- `k`: 2D transverse momentum vector in cartesian coordinates
- `μ`: Regulator for integrand (default: 0.00)
- `solver`: Integration strategy. Options: :cuhre, :vegas, :divonne, :suave. Default is :vegas

Returns
- `result::Vector{Float64}`: Array [re, im] containing real and imaginary parts of the gluon Sivers function f_{1T}^{⊥ g}(x,k)
- `err::Vector{Float64}`: Array [err_re, err_im] containing error estimates for real and imaginary parts
- `prob::Vector{Float64}`: Array [prob_re, prob_im] containing probability estimates for each component
- `neval::Int64`: Number of integrand evaluations
- `fail::Int32`: Integration failure flag (0 = success)
- `nregions::Int32`: Number of subregions used in the integration

Notes
- Assumes the 2D k_T vector is [kx, 0]
- For k not along x, one would need to compute both (s01, s02) = (1, -1) and (-1, 1) 
  and sum them similar to `f2_form_factor`
- Result is real but we keep the imaginary part for consistency
"""
function gluon_sivers(k::AbstractVector{<:Real}; μ::Real=0.00, solver::Symbol=:vegas)
    # Spin flip
    s01 = 1
    s02 = -1
    # Zero momentum transfer
    Δ = hp.vec2(0.0, 0.0)

    odderon_dist, err, prob, neval, fail, nregions  = odderon_distribution(s01, s02, Δ, k; μ=μ, solver=solver)
    # 1 / k^2 partially cancels with definition of Sivers function
    kabs = sqrt(hp.sqnorm2(k))
    prf = 8 * M_N * NC / π * ALPHA_S^2 / kabs
    result = prf * odderon_dist
    err .*= abs(prf)
    return result, err, prob, neval, fail, nregions
end

# ======================
end # module Odderon
