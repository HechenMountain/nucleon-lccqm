module Sivers
# Main module for computing the gluon Sivers function
# Parameters are defined in parameters.jl
# Helpers are defined in helpers.jl

# ======================
# Imports
# ======================
# Integration
using Cuba

const BasePath = @__DIR__
const ParametersPath = joinpath(BasePath,"parameters.jl")
const HelpersPath = joinpath(BasePath,"helpers.jl")
const LCQMPath = joinpath(BasePath,"lc-cqm.jl")

# Parameters handled separately
include(ParametersPath)
using .Parameters: params

# Light-cone constituent quark model wavefunctions
include(LCQMPath)
import .LC_CQM as wfs
# Load into namespace for export
normalize_wavefunction = wfs.normalize_wavefunction

# Helpers, coordinate transformations, etc.
include(HelpersPath)
import .Helpers as hp

# ======================
# Exports
# ======================

export  normalize_wavefunction,
        f1_form_factor,
        f2_form_factor,
        f_form_factor,
        cubic_color_correlator, 
        odderon_distribution,
        gluon_sivers

# ======================
# Constants
# ======================

# Parameters and SU(Nc) algebra set in parameters.jl
const αs = params.αs ;
const Nc = params.Nc ;
const mN = params.mN ;
const dabc2 = (Nc^2 - 4) * (Nc^2 - 1) / Nc ;

# ======================
# Form Factors
# ======================

"""
    f_form_factor(s01::Integer, s02::Integer, Δ::Vector{<:Real})

Compute the F-type form factor needed to generate F1 and F2.

# Arguments
- `s01, s02`: Spins of the ingoing/outgoing protons (each must be either +1 or -1)
- `Δ`: 2D momentum transfer vector in cartesian coordinates

# Returns
- `integral::Vector{Float64}`: Array [re, im] containing real and imaginary parts of the form factor
- `err::Vector{Float64}`: Array [err_re, err_im] containing error estimates for real and imaginary parts
- `prob::Vector{Float64}`: Array [prob_re, prob_im] containing probability estimates for each component
- `neval::Int64`: Number of integrand evaluations
- `fail::Int32`: Integration failure flag (0 = success)
- `nregions::Int32`: Number of subregions used in the integration

# Notes
- `norm` is set in `parameters.jl` and can be obtained from `normalize_wavefunction()`
- Spin sums are performed inside integrand, then `cuhre` is used once to integrate both real and imaginary parts
- Momenta must be in cartesian coordinates
"""
function f_form_factor(s01::Integer, s02::Integer, Δ::Vector{<:Real})
    eu, ed = 2/3, -1/3
    charges = (eu,eu,ed)
    function integrand(x,f)
        # Regulate endpoint singularities
        x = hp.regulate_cuba(x)
        (x1, x2, x3), d2x = hp.cuba_to_parton_x(x[1:2])
        r1, ϕ1, d2k1 = hp.cuba_to_polar(x[3:4])    # k1
        r2, ϕ2, d2k2 = hp.cuba_to_polar(x[5:6])    # k2
        
        # Reconstruct cartesian momenta from polar coordinates
        k1 = hp.polar_to_cartesian([r1, ϕ1])
        k2 = hp.polar_to_cartesian([r2, ϕ2])
        k3 = - (k1 + k2)  # Enforce transverse momentum conservation

        # Jacobian
        d4k = d2k1 * d2k2

        total = 0
        wf1 = wfs.compute_wavefunction(s01, k1, k2, k3, x1, x2, x3)
        # Sum over charge contributions
        for (i,q) in enumerate(charges)
            k1prime = k1 - x1 * Δ + hp.δ(i,1) * Δ
            k2prime = k2 - x2 * Δ + hp.δ(i,2) * Δ
            k3prime = k3 - x3 * Δ + hp.δ(i,3) * Δ
            wf2 = wfs.compute_wavefunction(s02, k1prime, k2prime, k3prime, x1, x2, x3)
            # Sum over spin contributions
            total += q * wfs.spin_sum(wf1, wf2)
        end
        res = total * d4k * d2x
        f[1] = real(res)
        f[2] = imag(res)
    end
    # Call cuhre with ncomp=2 to track real and imaginary parts separately
    integral, err, prob, neval, fail, nregions = cuhre(integrand, 6, 2; maxevals=10_000_000)
    # Apply prefactors to integration results
    norm = params.norm
    prf = 3 / (4π)^2 / (2π)^4 * norm^2
    integral .*= prf
    err .*= prf
    return integral, err, prob, neval, fail, nregions
end

"""
    f1_form_factor(Δ::Vector{<:Real})

Compute the F1 form factor.

# Arguments
- `Δ`: 2D momentum transfer vector in cartesian coordinates

# Returns
- `result::Vector{Float64}`: Array [re, im] containing real and imaginary parts of the F1 form factor
- `err::Vector{Float64}`: Array [err_re, err_im] containing error estimates for real and imaginary parts
- `prob::Vector{Float64}`: Array [prob_re, prob_im] containing probability estimates for each component
- `neval::Int64`: Number of integrand evaluations
- `fail::Int32`: Integration failure flag (0 = success)
- `nregions::Int32`: Number of subregions used in the integration

# Notes
- `norm` is set in `parameters.jl` and can be obtained from `normalize_wavefunction()`
- Spin sums are performed inside integrand, then `cuhre` is used once to integrate both real and imaginary parts
- Momenta must be in cartesian coordinates
"""
function f1_form_factor(Δ::Vector{<:Real})
    result, err, prob, neval, fail, nregions = f_form_factor(1, 1, Δ)
    return result, err, prob, neval, fail, nregions
end

"""
    f2_form_factor(Δ::Vector{<:Real})

Compute the F2 form factor.

# Arguments
- `Δ`: 2D momentum transfer vector in cartesian coordinates

# Returns
- `integral::Vector{Float64}`: Array [re, im] containing real and imaginary parts of the F2 form factor
- `err::Vector{Float64}`: Array [err_re, err_im] containing error estimates for real and imaginary parts
- `prob::Vector{Float64}`: Array [prob_re, prob_im] containing probability estimates for each component
- `neval::Int64`: Number of integrand evaluations
- `fail::Int32`: Integration failure flag (0 = success)
- `nregions::Int32`: Number of subregions used in the integration

# Notes
- `norm` is set in `parameters.jl` and can be obtained from `normalize_wavefunction()`
- Spin sums are performed inside integrand, then `cuhre` is used once to integrate both real and imaginary parts
- Momenta must be in cartesian coordinates
"""
function f2_form_factor(Δ::Vector{<:Real})
    ΔL, ΔR = complex(Δ[1], -Δ[2]) , complex(Δ[1], Δ[2]) 
    Δ2 = sum(Δ.^2)
    # Notation in notes reversed: Lambda', Lambda = s02, s01
    fdu, err_du, prob, neval, fail, nregions = f_form_factor(1, -1, Δ)
    fud, err_ud, = f_form_factor(-1, 1, Δ)

    # Calculate real and imaginary parts separately
    # fdu and fud are arrays with [re, im] parts
    prf = mN / Δ2
    result_re = mN^2 / Δ2 * (ΔL / mN * fdu[1] - ΔR / mN * fud[1])
    result_im = mN^2 / Δ2 * (ΔL / mN * fdu[2] - ΔR / mN * fud[2])
    err_re = mN / sqrt(Δ2) * sqrt(err_du[1]^2 + err_ud[1]^2)
    err_im = mN / sqrt(Δ2) * sqrt(err_du[2]^2 + err_ud[2]^2)
    
    return [result_re, result_im], [err_re, err_im], prob, neval, fail, nregions
end

# ======================
# Distributions
# ======================

"""
    cubic_color_correlator(s01::Integer,s02::Integer,
                           x1::Real, x2::Real, x3::Real,
                           q1::Vector{<:Real}, q2::Vector{<:Real}, q3::Vector{<:Real},
                           k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real})

Compute the unintegrated cubic color correlator by summing one-, two-, and three-body contributions.

# Arguments
- `s01, s02`: Spins of the ingoing/outgoing protons (each must be either +1 or -1)
- `x1, x2, x3`: Parton x values satisfying x1 + x2 + x3 = 1
- `q1, q2, q3`: Eikonal momenta (2D cartesian vectors)
- `k1, k2, k3`: Transverse momenta (2D cartesian vectors)

# Returns
- `ccc::ComplexF64`: 'Value of the cubic color correlator for the given spin configuration and kinematics

# Notes
- This is G_3ΛΛ' stripped of the integrals and factor of 1/(4π)^2/(2π)^2 in the draft
- Momenta must be in cartesian coordinates
"""
function cubic_color_correlator(s01::Integer,s02::Integer,
                                x1::Real, x2::Real, x3::Real,
                                q1::Vector{<:Real}, q2::Vector{<:Real}, q3::Vector{<:Real},
                                k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real})
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
    wf1 = wfs.compute_wavefunction(s01, k1, k2, k3, x1, x2, x3)

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
        wf2 = wfs.compute_wavefunction(s02,k1prime,k2prime,k3prime,x1,x2,x3)
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
        wf2 = wfs.compute_wavefunction(s02, k1prime, k2prime, k3prime, x1, x2, x3)
        # Perform spin sum
        ccc += wfs.spin_sum(wf1, wf2)
    end
    # Three-body
    for j1 in 1:3, j2 in 1:3, j3 in 1:3
        if j1 == j2 || j1 == j3 || j2 == j3
            continue
        end
        k1prime = k1prime0 - three_body_kin(1, j1, j2, j3, q1, q2, q3)
        k2prime = k2prime0 - three_body_kin(2, j1, j2, j3, q1, q2, q3)
        k3prime = k3prime0 - three_body_kin(3, j1, j2, j3, q1, q2, q3)
        wf2 = wfs.compute_wavefunction(s02, k1prime, k2prime, k3prime, x1, x2, x3)
        # Perform spin sum
        ccc += wfs.spin_sum(wf1, wf2)
    end
    return ccc
end

"""
    integrate_cubic_color_correlator(s01::Integer,s02::Integer,
                           x1::Real, x2::Real, x3::Real,
                           q1::Vector{<:Real}, q2::Vector{<:Real}, q3::Vector{<:Real})

Compute the integrated cubic color correlator.

# Arguments
- `s01, s02`: Spins of the ingoing/outgoing protons (each must be either +1 or -1)
- `q1, q2, q3`: Eikonal momenta (2D cartesian vectors)
- `solver`: Integration strategy (default: "cuhre", options: "cuhre", "vegas", "divonne", "suave")

# Returns
- ToDo

# Notes
- This is G_3ΛΛ' including the integrals to run some checks.
- Momenta must be in cartesian coordinates
"""
function integrate_cubic_color_correlator(s01::Integer,s02::Integer,
                                          q1::Vector{<:Real}, q2::Vector{<:Real}, q3::Vector{<:Real};
                                          solver::String="vegas")
    sol =   solver == "cuhre" ? cuhre :
            solver == "vegas" ? vegas :
            solver == "suave" ? suave :
            solver == "divonne" ? divonne :
            throw(ArgumentError("solver must be one of: cuhre, vegas, divonne, suave"))

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
        k1 = hp.polar_to_cartesian([r1, ϕ1])
        k2 = hp.polar_to_cartesian([r2, ϕ2])
        k3 = - (k1 + k2)  # Enforce transverse momentum conservation
        d4k = d2k1 * d2k2

        # Jacobian
        d6x = d2x * d4k # 2 + 4 = 6d integral

        ccc = cubic_color_correlator(s01, s02, x1, x2, x3, q1, q2, q3, k1, k2, k3)
        ccc *=  d6x

        f[1] = real(ccc)
        f[2] = imag(ccc)
    end
    integral, err, prob, neval, fail, nregions = sol(integrand, 6, 2; maxevals=10_000_000)
    # Prefactors 
    norm = params.norm
    prf = norm^2
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
                           x1::Real, x2::Real, x3::Real,
                           q1::Vector{<:Real}, q2::Vector{<:Real}, q3::Vector{<:Real}
                           r::Vector{<:Real}, solver::String="vegas")

Compute the Fourier transform of the integrated cubic color correlator.

# Arguments
- `s01, s02`: Spins of the ingoing/outgoing protons (each must be either +1 or -1)
- `r`: 2D transverse coordinate vector in cartesian coordinates
- `solver`: Integration strategy (default: "cuhre", options: "cuhre", "vegas", "divonne", "suave")

# Returns
- ToDo

# Notes
- This is G_3ΛΛ' including the integrals to run some checks.
- Momenta must be in cartesian coordinates
"""
function ft_cubic_color_correlator(s01::Integer,s02::Integer,
                                   r::Vector{<:Real}; solver::String="vegas")
    sol =   solver == "cuhre" ? cuhre :
            solver == "vegas" ? vegas :
            solver == "suave" ? suave :
            solver == "divonne" ? divonne :
            throw(ArgumentError("solver must be one of: cuhre, vegas, divonne, suave"))

    function integrand(x,f)
        # Regulate endpoint singularities
        x = hp.regulate_cuba(x)
        # Transform [0,1]^8 cuba samples to physical variables
        # Parton-x
        (x1, x2, x3), d2x = hp.cuba_to_parton_x(x[1:2])
        # Momenta
        r1, ϕ1, d2k1 = hp.cuba_to_polar(x[3:4])     # k1
        r2, ϕ2, d2k2 = hp.cuba_to_polar(x[5:6])     # k2
        r3, ϕ3, d2Δ = hp.cuba_to_polar(x[7:8])      # Δ
    
        # Reconstruct cartesian momenta from polar coordinates
        k1 = hp.polar_to_cartesian([r1, ϕ1])
        k2 = hp.polar_to_cartesian([r2, ϕ2])
        k3 = - (k1 + k2)    # Enforce transverse momentum conservation
        d4k = d2k1 * d2k2

        Δ = hp.polar_to_cartesian([r3, ϕ3])
        # Assume q_12 = q_23 = 0 
        q12, q23 = [0,0], [0,0]
        q1, q2, q3 = (2 * q12 + q23 - Δ) / 3, (- q12 + q23 - Δ) / 3, - (q12 + 2 * q23 + Δ) / 3
        # Jacobian
        d8x = d2x * d4k * d2Δ   # 2 + 4 + 2 = 8d integral

        ccc = cubic_color_correlator(s01, s02, x1, x2, x3, q1, q2, q3, k1, k2, k3)
        # Fourier factor
        ft_factor = exp(-im * Δ'r)
        res = ccc * ft_factor * d8x

        f[1] = real(res)
        f[2] = imag(res)
    end
    integral, err, prob, neval, fail, nregions = sol(integrand, 8, 2; maxevals=100_000_000)
    # Prefactors 
    norm = params.norm
    prf = norm^2
    # π factors from integration:
    # Deltas have (2π)^2 * 4π in front. 
    # Every integration over k gives 1 / (2π)^2 -> 1 / (2π)^(2 * 3)
    # Every integration over x gives 1 / (4π)
    prf /= (4π)^2 * (2π)^6
    integral .*= prf
    err .*= abs(prf)
    return integral, err, prob, neval, fail, nregions 
end

"""
    odderon_distribution(s01::Integer,s02::Integer,
                         k::Vector{<:Real},Δ::Vector{<:Real};
                         μ::Real=0.00,solver::String="vegas")

Compute the Odderon distribution O * k^2 for momentum transfer k and Δ.

# Arguments
- `s01, s02`: Spin of ingoing/outgoing proton (each must be either +1 or -1)
- `k`: 2D transverse momentum transfer vector in cartesian coordinates
- `Δ`: 2D momentum transfer vector in cartesian coordinates
- `μ`: Regulator for integrand (default: 0.00)
- `solver`: Integration strategy (default: "vegas", options: "cuhre", "vegas", "divonne", "suave")

# Returns
- `integral::Vector{Float64}`: Array [re, im] containing real and imaginary parts of O(k,Δ) * k^2
- `err::Vector{Float64}`: Array [err_re, err_im] containing error estimates for real and imaginary parts
- `prob::Vector{Float64}`: Array [prob_re, prob_im] containing probability estimates for each component
- `neval::Int64`: Number of integrand evaluations
- `fail::Int32`: Integration failure flag (0 = success)
- `nregions::Int32`: Number of subregions used in the integration

# Notes
- Currently only valid for Δ = [0,0], corresponds to OΛΛ'(k,Δ) in the draft
- Factor 1/k^2 is omitted here and added later in the Sivers function definition
- Momenta must be in cartesian coordinates
- `norm` is set in `parameters.jl`, obtainable via `normalize_wavefunction()`
- Result is generally complex; for k_y = 0 it is real
- We enforce the symmetry in the k integration to get a simpler integrand
"""
function odderon_distribution(s01::Integer, s02::Integer,
                              k::Vector{<:Real}, Δ::Vector{<:Real};
                              μ::Real=0.00, solver::String="vegas")
    if !iszero(Δ)
        throw(ArgumentError("Implementation currently only for vanishing Δ."))
    end
    
    sol =   solver == "cuhre" ? cuhre :
            solver == "vegas" ? vegas :
            solver == "suave" ? suave :
            solver == "divonne" ? divonne :
            throw(ArgumentError("solver must be one of: cuhre, vegas, divonne, suave"))

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
        k1 = hp.polar_to_cartesian([r1, ϕ1])
        k2 = hp.polar_to_cartesian([r2, ϕ2])
        k3 = - (k1 + k2)  # Enforce transverse momentum conservation
        d4k = d2k1 * d2k2

        q2 = hp.polar_to_cartesian([r3, ϕ3])
        # Jacobian
        d8x = d2x * d4k * d2q2 # 2 + 4 + 2 = 8d integral
        μ2 = μ^2 # Regulator squared
        total = 0
        # Simplified integrand for spin-flip
        if s01 != s02
            for s in (+1,-1)
                # Flip momenta to project out Sivers function
                q1, q2, q3 = s * k, s * q2, - s * (k + q2)
                ccc = cubic_color_correlator(s01, s02, x1, x2, x3, q1, q2, q3, k1, k2, k3)
                total += s * ccc
            end
            # Regenerate initial q2
            q2 = hp.polar_to_cartesian([r3, ϕ3])
            q3 = k + q2
            q22, q32 = sum(q2.^2), sum(q3.^2)
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
            q22 = sum(q2.^2)
            # Add regulator
            q22 += μ2
            for s in (+1,-1)
                q1, q3 = s * k, - (s * k + q2)
                ccc = cubic_color_correlator(s01, s02, x1, x2, x3, q1, q2, q3, k1, k2, k3)
                q32 = sum(q3.^2)
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
    norm = params.norm
    # Factor of g^6 = 1, for simplicity, treated later on
    prf = - dabc2 / Nc / 32 * norm^2
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
    gluon_sivers(k::Real;μ::Real=0.00,solver::String="vegas")

Compute the gluon Sivers function for momentum transfer k.

# Arguments
- `k`: Modulus of transverse momentum transfer
- `μ`: Regulator for integrand (default: 0.01)
- `solver`: Integration strategy (default: "vegas", options: "cuhre", "vegas", "divonne", "suave")

# Returns
- `result::Vector{Float64}`: Array [re, im] containing real and imaginary parts of the gluon Sivers function f_{1T}^{⊥ g}(x,k)
- `err::Vector{Float64}`: Array [err_re, err_im] containing error estimates for real and imaginary parts
- `prob::Vector{Float64}`: Array [prob_re, prob_im] containing probability estimates for each component
- `neval::Int64`: Number of integrand evaluations
- `fail::Int32`: Integration failure flag (0 = success)
- `nregions::Int32`: Number of subregions used in the integration

# Notes
- Assumes the 2D k_T vector is [kx, 0]
- For k not along x, one would need to compute both (s01, s02) = (1, -1) and (-1, 1) 
  and sum them similar to `f2_form_factor`
- Result is real but we keep the imaginary part for consistency
"""
function gluon_sivers(k::Real; μ::Real=0.00, solver::String="vegas")
    # Spin flip
    s01 = 1
    s02 = -1
    # Zero momentum transfer
    Δ = [0,0]

    odderon_dist, err, prob, neval, fail, nregions  = odderon_distribution(s01, s02, [k,0], Δ; μ=μ, solver=solver)
    # 1 / k^2 partially cancels with definition of Sivers function
    prf = 8 * mN * Nc / π * αs^2 / k
    result = prf * odderon_dist
    err .*= abs(prf)
    return result, err, prob, neval, fail, nregions
end

# ======================
end # module