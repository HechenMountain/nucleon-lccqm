"""
    EMFormFactors

Module for computing electromagnetic form factors F1 and F2
in the light-cone constituent quark model.

## Main exports
- `f_form_factor`: general F-type form factor
- `f1_form_factor`: Dirac form factor F1
- `f2_form_factor`: Pauli form factor F2
"""
module EMFormFactors

# ======================
# Imports
# ======================

using Cuba

# Import parameters from parent module
import ..NORM, ..M_N

# Import sibling modules
import ..Helpers as hp
import ..Wavefunctions as wfs

# ======================
# Exports
# ======================

export f_form_factor,
       f1_form_factor,
       f2_form_factor

# ======================
# Form Factors
# ======================

"""
    f_form_factor(s01::Integer, s02::Integer, Δ::AbstractVector{<:Real})

Compute the F-type form factor needed to generate F1 and F2.

Arguments
- `s01, s02`: Spins of the ingoing/outgoing protons (each must be either +1 or -1)
- `Δ`: 2D momentum transfer vector in cartesian coordinates

Returns
- `integral::Vector{Float64}`: Array [re, im] containing real and imaginary parts of the form factor
- `err::Vector{Float64}`: Array [err_re, err_im] containing error estimates for real and imaginary parts
- `prob::Vector{Float64}`: Array [prob_re, prob_im] containing probability estimates for each component
- `neval::Int64`: Number of integrand evaluations
- `fail::Int32`: Integration failure flag (0 = success)
- `nregions::Int32`: Number of subregions used in the integration

Notes
- `NORM` is set in `parameters.jl` and can be obtained from `normalize_wavefunction()`
- Spin sums are performed inside integrand, then `cuhre` is used once to integrate both real and imaginary parts
- Momenta must be in cartesian coordinates
"""
function f_form_factor(s01::Integer, s02::Integer, Δ::AbstractVector{<:Real})
    eu, ed = 2/3, -1/3
    charges = (eu,eu,ed)
    function integrand(x,f)
        # Regulate endpoint singularities
        x = hp.regulate_cuba(x)
        (x1, x2, x3), d2x = hp.cuba_to_parton_x(x[1:2])
        r1, ϕ1, d2k1 = hp.cuba_to_polar(x[3:4])    # k1
        r2, ϕ2, d2k2 = hp.cuba_to_polar(x[5:6])    # k2
        
        # Reconstruct cartesian momenta from polar coordinates
        k1 = hp.polar_to_cartesian(hp.vec2(r1, ϕ1))
        k2 = hp.polar_to_cartesian(hp.vec2(r2, ϕ2))
        k3 = - (k1 + k2)  # Enforce transverse momentum conservation

        # Jacobian
        d4k = d2k1 * d2k2

        total = 0
        wf1 = wfs.compute_wavefunction(s01, x1, x2, x3, k1, k2, k3)
        # Sum over charge contributions
        for (i,q) in enumerate(charges)
            k1prime = k1 - x1 * Δ + hp.δ(i,1) * Δ
            k2prime = k2 - x2 * Δ + hp.δ(i,2) * Δ
            k3prime = k3 - x3 * Δ + hp.δ(i,3) * Δ
            wf2 = wfs.compute_wavefunction(s02, x1, x2, x3, k1prime, k2prime, k3prime)
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
    wf_norm = NORM
    prf = 3 / (4π)^2 / (2π)^4 * wf_norm^2
    integral .*= prf
    err .*= prf
    return integral, err, prob, neval, fail, nregions
end

"""
    f1_form_factor(Δ::AbstractVector{<:Real})

Compute the F1 form factor.

Arguments
- `Δ`: 2D momentum transfer vector in cartesian coordinates

Returns
- `result::Vector{Float64}`: Array [re, im] containing real and imaginary parts of the F1 form factor
- `err::Vector{Float64}`: Array [err_re, err_im] containing error estimates for real and imaginary parts
- `prob::Vector{Float64}`: Array [prob_re, prob_im] containing probability estimates for each component
- `neval::Int64`: Number of integrand evaluations
- `fail::Int32`: Integration failure flag (0 = success)
- `nregions::Int32`: Number of subregions used in the integration

Notes
- `NORM` is set in `parameters.jl` and can be obtained from `normalize_wavefunction()`
- Spin sums are performed inside integrand, then `cuhre` is used once to integrate both real and imaginary parts
- Momenta must be in cartesian coordinates
"""
function f1_form_factor(Δ::AbstractVector{<:Real})
    result, err, prob, neval, fail, nregions = f_form_factor(1, 1, Δ)
    return result, err, prob, neval, fail, nregions
end

"""
    f2_form_factor(Δ::AbstractVector{<:Real})

Compute the F2 form factor.

Arguments
- `Δ`: 2D momentum transfer vector in cartesian coordinates

Returns
- `integral::Vector{Float64}`: Array [re, im] containing real and imaginary parts of the F2 form factor
- `err::Vector{Float64}`: Array [err_re, err_im] containing error estimates for real and imaginary parts
- `prob::Vector{Float64}`: Array [prob_re, prob_im] containing probability estimates for each component
- `neval::Int64`: Number of integrand evaluations
- `fail::Int32`: Integration failure flag (0 = success)
- `nregions::Int32`: Number of subregions used in the integration

Notes
- `NORM` is set in `parameters.jl` and can be obtained from `normalize_wavefunction()`
- Spin sums are performed inside integrand, then `cuhre` is used once to integrate both real and imaginary parts
- Momenta must be in cartesian coordinates
"""
function f2_form_factor(Δ::AbstractVector{<:Real})
    ΔL, ΔR = complex(Δ[1], -Δ[2]) , complex(Δ[1], Δ[2]) 
    Δ2 = hp.sqnorm2(Δ)
    # Notation in notes reversed: Lambda', Lambda = s02, s01
    fdu, err_du, prob, neval, fail, nregions = f_form_factor(1, -1, Δ)
    fud, err_ud, = f_form_factor(-1, 1, Δ)

    # Compose full complex form factors and contract with ΔL/ΔR
    fdu_c = complex(fdu[1], fdu[2])
    fud_c = complex(fud[1], fud[2])
    full = M_N^2 / Δ2 * (ΔL / M_N * fdu_c - ΔR / M_N * fud_c)
    result_re = real(full)
    result_im = imag(full)
    err_re = M_N / sqrt(Δ2) * sqrt(err_du[1]^2 + err_ud[1]^2)
    err_im = M_N / sqrt(Δ2) * sqrt(err_du[2]^2 + err_ud[2]^2)

    return [result_re, result_im], [err_re, err_im], prob, neval, fail, nregions
end

# ======================
end # module EMFormFactors
