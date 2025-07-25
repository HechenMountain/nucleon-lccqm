module Sivers

using Base: pi
using Printf
using LinearAlgebra
using PyPlot
using Cuba

# Parallelization
using Base.Threads

# Progress bar
using ProgressMeter
const Atomic = Base.Threads.Atomic
const atomic_add! = Base.Threads.atomic_add!

# Parameters handled separately
include("/mnt/c/Users/flori/Documents/PostDoc/Jupyter/Julia/sivers/parameters.jl")
using .parameters: params

# Color algebra
include("/mnt/c/Users/flori/Documents/PostDoc/Jupyter/Julia/sivers/GellMann.jl")
using .GellMann

export  f1_form_factor, f1_form_factor_table,
        cubic_color_corellator, odderon_distribution, gluon_sivers
        # SPIN_MAP,kronecker_delta, normalize_wavefunction,
        # momentum_space_wavefunction, spin_wavefunction,
        # baryon_wavefunction,precompute_wavefunction, spin_sum

# Parameters and SU(Nc) algebra set in parameters.jl
alpha_s = params.alpha_s
Nc = params.Nc
mN = params.mN
dabc2 = (Nc^2 - 4) * (Nc^2 - 1) / Nc 

###############
### Helpers ###
###############

# All possible proton and constituent quark spin configurations
# Eq.(22) and (23) in the draft
# Maps spin tuple with externally assigned 
# kinematical parameters to expression
const SPIN_MAP = Dict{NTuple{4, Int}, Function}(
    # s0 = +1
    (+1, +1, +1, -1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) ->  2a1*a2*a3 + a1*k2L*k3R + k1L*a2*k3R,
    (+1, +1, -1, +1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> -a1*a2*a3 + k1L*k2R*a3 - 2a1*k2R*k3L,
    (+1, -1, +1, +1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> -a1*a2*a3 + k1R*k2L*a3 - 2k1R*a2*k3L,
    (+1, +1, -1, -1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) ->  a1*a2*k3R - 2a1*k2R*a3 - k1L*k2R*k3R,
    (+1, -1, +1, -1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) ->  a1*a2*k3R - 2k1R*a2*a3 - k1R*k2L*k3R,
    (+1, -1, -1, +1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) ->  a1*k2R*a3 + k1R*a2*a3 + 2k1R*k2R*k3L,
    (+1, +1, +1, +1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> -a1*k2L*a3 - k1L*a2*a3 + 2a1*a2*k3L,
    (+1, -1, -1, -1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> -a1*k2R*k3R - k1R*a2*k3R + 2k1R*k2R*a3,
    # s0 = -1
    (-1, -1, -1, +1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> - (2a1*a2*a3 + a1*k2R*k3L + k1R*a2*k3L),
    (-1, -1, +1, -1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> - (-a1*a2*a3 + k1R*k2L*a3 - 2a1*k2L*k3R),
    (-1, +1, -1, -1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> - (-a1*a2*a3 + k1L*k2R*a3 - 2k1L*a2*k3R),
    (-1, -1, +1, +1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> - (a1*a2*k3L - 2a1*k2L*a3 - k1R*k2L*k3L),
    (-1, +1, -1, +1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> - (a1*a2*k3L - 2k1L*a2*a3 - k1L*k2R*k3L),
    (-1, +1, +1, -1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> - (a1*k2L*a3 + k1L*a2*a3 + 2k1L*k2L*k3R),
    (-1, -1, -1, -1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> - (-a1*k2R*a3 - k1R*a2*a3 + 2a1*a2*k3R),
    (-1, +1, +1, +1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> - (-a1*k2L*k3L - k1L*a2*k3L + 2k1L*k2L*a3),
);

# δ_ij
kronecker_delta(a, b) = a == b ? 1 : 0

# Map spin -1 (+1) to index 1 (2)
spin_index(s) = (s == -1) ? 1 : 2

#####################
### Wavefunctions ###
#####################

"""
    spin_wavefunction(s0,s1,s2,s3,k1, k2, k3, x1, x2, x3)

Spin dependence of baryon wavefunction obtained from Clebsch-Gordan coefficients.
0 refers to proton, 1-3 are valence quark indices
# Arguments
- `s0, s1, s2, s3::Integer`: Spin indices of proton and valence quarks. Either +1 or -1
- `k1, k2, k3::Vector{<:Real}`: Momenta of valence quarks
    - 2d real vectors
- `x1, x2, x3::Real`: Parton x of valence quarks
    - Must be in [0,1]

# Returns
Value of spinor wavefunction
"""
function spin_wavefunction(s0::Integer,s1::Integer,s2::Integer,s3::Integer,k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real}, x1::Real, x2::Real, x3::Real)
    # Parameters
    mq = params.mq

    if !all(s -> s == 1 || s == -1, (s0,s1,s2,s3))
        error("Invalid spin configuration:  ($(s0), $(s1), $(s2), $(s3)). Each value must be +1 or -1.")
    end
    k12, k22, k32 = sum(k1.^2), sum(k2.^2), sum(k3.^2)
    mq2 = mq^2
    m02 = (k12 + mq2)/x1 + (k22 + mq2)/x2 + (k32 + mq2)/x3
    m0 = sqrt(m02)
    a1 = mq + x1 * m0
    a2 = mq + x2 * m0
    a3 = mq + x3 * m0

    k1L, k1R = complex(k1[1],-k1[2]),complex(k1[1],k1[2])
    k2L, k2R = complex(k2[1],-k2[2]),complex(k2[1],k2[2])
    k3L, k3R = complex(k3[1],-k3[2]),complex(k3[1],k3[2])

    n1, n2, n3 = k12 + a1^2, k22 + a2^2, k32 + a3^2
    norm = 1/sqrt(6*n1*n2*n3)
    wf = norm * SPIN_MAP[(s0,s1,s2,s3)](a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R)
    return wf
end

"""
    momentum_space_wavefunction(k1, k2, k3, x1, x2, x3)

Momentum dependence of baryon wavefunction
# Arguments
- `k1, k2, k3::Vector{<:Real}`: Momenta of valence quarks
    - 2d real vectors
- `x1, x2, x3::Real`: Parton x of valence quarks
    - Must be in [0,1]

# Returns
Value of momentum space wavefunction
"""
function momentum_space_wavefunction(k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real}, x1::Real, x2::Real, x3::Real)
    # Parameters
    mq = params.mq
    β = params.β

    mq2 = mq^2
    m02 = (sum(k1.^2) + mq2) / x1 + (sum(k2.^2) + mq2) / x2 + (sum(k3.^2) + mq2) / x3
    return exp(-m02 / (2 * β^2))
end

"""
    baryon_wavefunction(s0,s1,s2,s3,k1, k2, k3, x1, x2, x3)

Compute product of spinor and momentum space wave function.
0 refers to proton, 1-3 are valence quark indices
# Arguments
- `s0, s1, s2, s3::Integer`: Spin indices of proton and valence quarks. 
    - Must be either +1 or -1
- `k1, k2, k3::Vector{<:Real}`: Momenta of valence quarks
    - 2d real vectors
- `x1, x2, x3::Real`: Parton x of valence quarks
    - Must be in [0,1]

# Returns
Value of baryon wavefunction

# Notes
norm is set in parameters.jl and can be obtained from normalize_wavefunction(s0)
"""
function baryon_wavefunction(s0::Integer,s1::Integer,s2::Integer,s3::Integer,k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real}, x1::Real, x2::Real, x3::Real)
    # Parameters
    norm = params.norm

    ms_wf =  momentum_space_wavefunction(k1, k2, k3, x1, x2, x3)
    spin_wf = spin_wavefunction(s0,s1,s2,s3,k1,k2,k3,x1,x2,x3)
    wf = norm * ms_wf * spin_wf / sqrt(3)
    return wf
end

"""
    precompute_wavefunction(s,k1, k2, k3, x1, x2, x3)

Precompute wavefunction and write index combinatioons to array.

# Arguments
- `s::Integer`: Spin index of proton.
    - Must be either +1 or -1
- `k1, k2, k3::Vector{<:Real}`: Momenta of valence quarks
    - 2d real vectors
- `x1, x2, x3::Real`: Parton x of valence quarks
    - Must be in [0,1]

# Returns
Value of norm of baryon wavefunction. To be set in parameters.jl

# Notes
The indices 1 (-1) of the valence quarks are mapped to 1 (2) 
such that the wavefunction is accessed as e.g. wf[1,2,1]
"""
function precompute_wavefunction(s::Integer,k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real}, x1::Real, x2::Real, x3::Real)
    # initialize array with 2^3 spin configurations
    wf = Array{ComplexF64}(undef, 2, 2, 2)
    for s1 in (-1,1), s2 in (-1,1), s3 in (-1,1)
        i1 = spin_index(s1)
        i2 = spin_index(s2)
        i3 = spin_index(s3)
        wf[i1,i2,i3] = baryon_wavefunction(s, s1, s2, s3, k1, k2, k3, x1, x2, x3)
    end
    return wf
end

"""
    spin_sum(wf1,wf2)

Carries out the spin sum over 2 wavefunctions.

# Arguments
- `wf1,wf2::Array{ComplexF64, 3}`: Baryon wavefunctions
    - Must be array with 2^3 entries

# Returns
Value of ∑conj(wf2) * wf1

# Notes
The wavefunctions wf1, wf2 passed to this object need to be in array form
as generated by precompute_wavefunction.
"""
# Wrapper for spin sum
function spin_sum(wf1::Array{ComplexF64, 3},wf2::Array{ComplexF64, 3})
    total = zero(ComplexF64)
    # Sum over spins
    for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
        i1 = spin_index(s1)
        i2 = spin_index(s2)
        i3 = spin_index(s3)
        total += conj(wf2[i1,i2,i3]) * wf1[i1,i2,i3]
    end
    return total
end

"""
    normalize_wavefunction()

Normalize the baryon wave function with parameters defined in parameters.jl

    # Returns
Value of norm of baryon wavefunction. To be set in parameters.jl

# Notes
norm is set in parameters.jl and can be obtained from normalize_wavefunction(s0).
Summation is done inside the integrand such that cuhre is only called once.
There is a commented out version in the module where individual contributions
are integrated separately and then summed. This gives a slight speedup, might be
useful in the future.
"""
function normalize_wavefunction()
    function integrand(x,f)
        x1 = x[1]
        x2 = (1 - x1) * x[2]
        x3 = 1 - x1 - x2
        dx = (1-x1)
        # Cuba samples are [0,1]^n so we
        # transform to polar coordinates
        r1 = x[3] / (1 - x[3])  # r ∈ [0, ∞)
        ϕ1 = 2π * x[4]          # φ ∈ [0, 2π)
        # Same for k2
        r2 = x[5] / (1 - x[5])
        ϕ2 = 2π * x[6]
        
        # Reconstruct momenta in polar coordinates
        k1 = [r1 * cos(ϕ1), r1 * sin(ϕ1)]
        k2 = [r2 * cos(ϕ2), r2 * sin(ϕ2)]
        k3 = - (k1 + k2)  # Enforce transverse momentum conservation

        # Jacobian
        dk1 = r1/(1 - x[3])^2 # r1
        dk2 = r2/(1 - x[5])^2 # r2
        dϕ = (2π)^2 # dϕ1 * dϕ2
        dk = dk1 * dk2
        # Precompute momentum space wavefunction
        ms_wf = momentum_space_wavefunction(k1, k2, k3, x1, x2, x3)
        # Sum over spin contributions
        total = 0.0
        for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
            spin_wf = spin_wavefunction(1,s1,s2,s3,k1,k2,k3,x1,x2,x3) 
            wf = ms_wf * spin_wf
            total += abs2(wf)
        end
        f[1] =  total * dk * dϕ * dx
    end
    integral, err = cuhre(integrand, 6, 1, atol=1e-8, rtol=1e-6);
    # Multiply with prefactors from integration
    result = 1 / (4π)^2 / (2π)^4 * integral[1]
    # Return norm
    norm = 1 / sqrt(result)
    return norm
end

####################
### Form factors ###
####################

"""
    f1_form_factor(Δ)

Compute the F1 form factor
# Arguments
- `Δ::Vector{<:Real}`: Momentum transfer
    - 2d real vector

# Returns
Value of the F1 form factor for a given momentum transfer

# Notes
norm is set in parameters.jl and can be obtained from normalize_wavefunction(s0)
Spin sums etc. are performed inside integrand, then cuhre is used once.
"""
function f1_form_factor(Δ::Vector{<:Real})
    eu, ed = 2/3, -1/3
    charges = (eu,eu,ed)
    s01, s02 = 1, 1
    function integrand(x,f)
        x1 = x[1]
        x2 = (1 - x1) * x[2]
        x3 = 1 - x1 - x2
        dx = (1-x1)
        # Cuba samples are [0,1]^n so we
        # transform to polar coordinates
        r1 = x[3] / (1 - x[3])  # r ∈ [0, ∞)
        ϕ1 = 2π * x[4]          # φ ∈ [0, 2π)
        # Same for k2
        r2 = x[5] / (1 - x[5])
        ϕ2 = 2π * x[6]
        
        # Reconstruct momenta in polar coordinates
        k1 = [r1 * cos(ϕ1), r1 * sin(ϕ1)]
        k2 = [r2 * cos(ϕ2), r2 * sin(ϕ2)]

        k3 = - (k1 + k2)  # Enforce transverse momentum conservation

        # Jacobian
        dk1 = r1/(1 - x[3])^2 # r1
        dk2 = r2/(1 - x[5])^2 # r2
        dϕ = (2π)^2 # dϕ1 * dϕ2
        dk = dk1 * dk2

        total = 0
        wf1 = precompute_wavefunction(s01,k1,k2,k3,x1,x2,x3)
        # Sum over charge contributions
        for (i,q) in enumerate(charges)
            k1prime = k1 - x1 * Δ + kronecker_delta(i,1) * Δ
            k2prime = k2 - x2 * Δ + kronecker_delta(i,2) * Δ
            k3prime = k3 - x3 * Δ + kronecker_delta(i,3) * Δ
            wf2 = precompute_wavefunction(s02,k1prime,k2prime,k3prime,x1,x2,x3)
            # Sum over spin contributions
            total += q * spin_sum(wf1,wf2)
        end
        res = total * dk * dϕ * dx
        # if abs(imag(res)) > 1e-10
        #     error("Imaginary part in integrand: imag = $(imag(res))")
        # end
        f[1] = real(res)
    end
    integral, err = cuhre(integrand, 6, 1, atol=1e-12, rtol=1e-10);
    # Multiply with prefactors from integration
    result =  3 / (4π)^2 / (2π)^4 * integral[1]
    return result
end

# """
#     f1_form_factor(Δ)

# Compute the F1 form factor
# # Arguments
# `Δ`: Momentum transfer; 2-vector

# # Returns
# Value of the F1 form factor for a given momentum transfer

# # Notes
# norm is set in parameters.jl and can be obtained from normalize_wavefunction(s0)
# Spin sums etc. are performed outside integrand, then results from cuhre are summed.
# """
# function f1_form_factor(Δ)
#     eu, ed = 2/3, -1/3
#     charges = (eu,eu,ed)
#     result = 0
#     # Sum over charge contributions
#     for (i,q) in enumerate(charges)
#         # Sum over spin contributions
#         for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
#             function integrand(x,f)
#                 x1 = x[1]
#                 x2 = (1 - x1) * x[2]
#                 x3 = 1 - x1 - x2
#                 dx = (1-x1)
#                 # Cuba samples are [0,1]^n so we
#                 # transform to polar coordinates
#                 r1 = x[3] / (1 - x[3])  # r ∈ [0, ∞)
#                 ϕ1 = 2π * x[4]          # φ ∈ [0, 2π)
#                 # Same for k2
#                 r2 = x[5] / (1 - x[5])
#                 ϕ2 = 2π * x[6]
                
#                 # Reconstruct momenta in polar coordinates
#                 k1 = [r1 * cos(ϕ1), r1 * sin(ϕ1)]
#                 k2 = [r2 * cos(ϕ2), r2 * sin(ϕ2)]

#                 k3 = - (k1 + k2)  # Enforce transverse momentum conservation

#                 # Jacobian
#                 dk1 = r1/(1 - x[3])^2 # r1
#                 dk2 = r2/(1 - x[5])^2 # r2
#                 dϕ = (2π)^2 # dϕ1 * dϕ2
#                 dk = dk1 * dk2
            
#                 k1prime = k1 - x1 * Δ + kronecker_delta(i,1) * Δ
#                 k2prime = k2 - x2 * Δ + kronecker_delta(i,2) * Δ
#                 k3prime = k3 - x3 * Δ + kronecker_delta(i,3) * Δ

#                 # Both wave functions have spin up
#                 wf1 = baryon_wavefunction(1,s1,s2,s3,k1,k2,k3,x1,x2,x3)
#                 wf2 = baryon_wavefunction(1,s1,s2,s3,k1prime,k2prime,k3prime,x1,x2,x3)
               
#                 res = q * conj(wf2) * wf1 * dk * dϕ * dx
#                 # if abs(imag(res)) > 1e-10
#                 #     error("Imaginary part in integrand: imag = $(imag(res))")
#                 # end
#                 f[1] = real(res)
#             end
#             integral, err = cuhre(integrand, 6, 1, atol=1e-12, rtol=1e-10);
#             result +=  integral[1]
#         end
#     end
#     result *= 3 / (4π) / (2π)^2
#     return result
# end

"""
    f1_form_factor_table(Δ_array)

Compute an array of F1 form factor values using f1_form_factor([Δ, 0.0])
where Δ is in Δ_array
# Arguments
- `Δ_array::Vector{<:Real}`: Array of momentum transfer as scalars.
    - Can have any length

# Returns
Values of the F1 form factor at the specified momentum transfers

# Notes
This function is parallelized. For parallelization to take effect one needs to
set export JULIA_NUM_THREADS=n, with n the number of threads before starting the jupyter server.
Didn't give a speed-up on my laptop.
"""
function f1_form_factor_table(Δ_array::Array)
    n = length(Δ_array)
    results = Vector{Float64}(undef, n)
    prog = Progress(n, desc="Computing F1 form factor in parallel...")
    counter = Atomic{Int}(0)

    @threads for i in 1:n
        Δ = Δ_array[i]
        results[i] = f1_form_factor([Δ, 0.0])
        val = atomic_add!(counter, 1)
        next!(prog; showvalues = [(:done, val)])
    end

    return results
end

#####################
### Distributions ###
#####################

"""
    cubic_color_corellator(s01,s02,q1,q2,q3,x)

Compute the unintegrated cubic color corellator by summing one-, two-,
and three-body contributions.
# Arguments
- `s01,s02::Integer`: Spins of the external protons
    - Must be either +1 or -1
- 'q1,q2,q3::Vector{<:Real}': Transverse momentum transfers
    - 2d real vectors
- 'x::Vector{<:Real}': [0,1]^6 cuba samples parametrizing k_perp integration
    - 6d real vectors with entries in [0,1]

# Returns
Value of the cubic color corellator for the given spin configuration and kinematics

# Notes
This is G_3ΛΛ' stripped of the integrals in the draft

# To do
Prefactors need to be checked, kinematics should be ok.
"""
function cubic_color_corellator(s01::Integer,s02::Integer,q1::Vector{<:Real},q2::Vector{<:Real},q3::Vector{<:Real},x::Vector{<:Real})
    # We define the kinematics here, the rest is equivalent
    # the factor of dabc from the sivers function is contained
    # in this expression to optimize the calls
    function one_body_kin(i,j123)
        # Momentum inflow [q1 + q2 + q3] at j123
        # i is k_prime (quark) index, j123 quark line with 
        # momentum inflow q1 + q2 + q3 from gluon.
        delta_kiprime =  kronecker_delta(i, j123) * (q1 + q2 + q3)
        return delta_kiprime
    end

    function two_body_kin(i,j12,j3,l)
        # Momentum inflow [q1 + q2,j12] [q3,j3] at j12 and j3
        # i is k_prime (quark) index, j12, j3 quark line with
        # momentum inflow q1 + q2 and q3 from gluon.
        # Addtional terms from permutations, so in total 3 contributions
        # which we distinguish by l
        if l == 1 # [q2 + q3,j12] [q1,j3]
            delta_kiprime =  kronecker_delta(i, j12) * (q2 + q3) + kronecker_delta(i, j3) * q1
        elseif l == 2 # [q1 + q3,j12] [q2,j3]
            delta_kiprime =  kronecker_delta(i, j12) * (q1 + q3) + kronecker_delta(i, j3) * q2
        elseif l == 3 # [q1 + q2,j12] [q3,j3]
            delta_kiprime =  kronecker_delta(i, j12) * (q1 + q2) + kronecker_delta(i, j3) * q3
        end
        return delta_kiprime
    end

    function three_body_kin(i,j1,j2,j3)
        # Momentum inflow [q1,j1] [q2,j2] [q3,j3]
        # i is k_prime (quark) index, j1, j2, j3 are gluons with momenta
        # q1, q2 and q3, respectively, attached to quark lines.
        delta_kiprime = kronecker_delta(i, j1) * q1 + kronecker_delta(i, j2) * q2 + kronecker_delta(i, j3) * q3
        return delta_kiprime
    end
    # Essentially as in f1_form_factor but with different kinematics.
    # We return just the integrand such that
    # cuhre is only called once in the end.

    # Δ determined from overall momentum conservation
    Δ = - (q1 + q2 + q3)

    # Cuba samples are [0,1]^n
    x1 = x[1]
    x2 = (1 - x1) * x[2]
    x3 = 1 - x1 - x2
    dx = (1-x1)
    # Transform to polar coordinates for k1
    r1 = x[3] / (1 - x[3])  # r ∈ [0, ∞)
    ϕ1 = 2π * x[4]          # φ ∈ [0, 2π)
    # Same for k2
    r2 = x[5] / (1 - x[5])
    ϕ2 = 2π * x[6]
    
    # Reconstruct momenta in polar coordinates
    k1 = [r1 * cos(ϕ1), r1 * sin(ϕ1)]
    k2 = [r2 * cos(ϕ2), r2 * sin(ϕ2)]
    k3 = - (k1 + k2)  # Enforce transverse momentum conservation

    # Precompute incoming baryon wavefunction
    wf1 = precompute_wavefunction(s01,k1, k2, k3, x1, x2, x3)

    # Jacobian
    dk1 = r1/(1 - x[3])^2 # r1
    dk2 = r2/(1 - x[5])^2 # r2
    dϕ = (2π)^2 # dϕ1 * dϕ2
    dk = dk1 * dk2

    # Precompute constant parts
    k1prime0 = k1 - x1 * Δ 
    k2prime0 = k2 - x2 * Δ 
    k3prime0 = k3 - x3 * Δ

    # Initialize outgoing wavefunction array
    total_wf2 = Array{ComplexF64}(undef, 2, 2, 2)
    fill!(total_wf2, ComplexF64(0.0, 0.0))

    # Sum over one-body, two-body and three-body kinematics
    # One-body
    for j123 in 1:3
        k1prime = k1prime0 - one_body_kin(1,j123)
        k2prime = k2prime0 - one_body_kin(2,j123)
        k3prime = k3prime0 - one_body_kin(3,j123)
        wf2 = precompute_wavefunction(s02,k1prime,k2prime,k3prime,x1,x2,x3)
        # Sum outgoing wavefunctions
        total_wf2 .+= wf2
    end
    # Two-body
    # Addtional terms from permutations, so we have an extra sum over k
    for l in 1:3, j12 in 1:3, j3 in 1:3
        if j12 == j3
            continue
        end
        k1prime = k1prime0 - two_body_kin(1,j12,j3,l)
        k2prime = k2prime0 - two_body_kin(2,j12,j3,l)
        k3prime = k3prime0 - two_body_kin(3,j12,j3,l)
        wf2 = precompute_wavefunction(s02,k1prime,k2prime,k3prime,x1,x2,x3)
        # Sum outgoing wavefunctions
        total_wf2 .-= .5 * wf2 
    end
    # Three-body
    for j1 in 1:3, j2 in 1:3, j3 in 1:3
        if j1 == j2 || j1 == j3 || j2 == j3
            continue
        end
        k1prime = k1prime0 - three_body_kin(1,j1,j2,j3)
        k2prime = k2prime0 - three_body_kin(2,j1,j2,j3)
        k3prime = k3prime0 - three_body_kin(3,j1,j2,j3)
        wf2 = precompute_wavefunction(s02,k1prime,k2prime,k3prime,x1,x2,x3)
        # Sum outgoing wavefunctions
        total_wf2 .+= wf2
    end
    # Perform spin sum once
    total = spin_sum(wf1,total_wf2)
    result = total * dk * dϕ * dx
    # Multiply with prefactors from integration
    result *= 1 / (4π)^2 / (2π)^2
    
    return result
end

"""
    odderon_distribution(k,Δ)

Compute the Odderon distribution O * k^2 for momentum transfer k and Δ
# Arguments
- `k::Vector{<:Real}`: Transverse momentum transfer
    - 2d real vector
- `Δ::Vector{<:Real}`: Total momentum transfer
    - 2d real vector

# Returns
Value of the the Odderon distribution times k^2 at k and Δ.

# Notes
For now this is expression is only valid for Δ = [0,0].
Corresponds to OΛΛ'(k,Δ) in the draft.
We drop the 1 / k^2 since it cancels with the corresponding
factor in the definition of the sivers function.

# To do
Prefactors need to be checked, kinematics should be ok.
"""
function odderon_distribution(s01::Integer,s02::Integer,k::Vector{<:Real},Δ::Vector{<:Real})
    if !iszero(Δ)
        throw(ArgumentError("Implementation currently only for vanishing Δ."))
    end
    function integrand(x,f)
        # 6d input for cubic_color_corellator
        x6 = x[1:6]
        # 2d input for q2 integral
        x2 = x[7:8]
        # Cuba samples are [0,1]^n so we
        # transform to polar coordinates
        r2 = x2[1] / (1 - x2[1])  # r ∈ [0, ∞)
        ϕ2 = 2π * x2[2]          # φ ∈ [0, 2π)
        
        # Reconstruct momentum in polar coordinates
        q2 = [r2 * cos(ϕ2), r2 * sin(ϕ2)]

        # Jacobian
        dq2 = r2 / (1 - x2[1])^2
        dϕ = (2π)

        total = 0
        for s in (+1,-1)
            q1 = s * k
            q3 = - s * k - q2
            # Denominator excluding q1^2 = k^2
            q32 = sum(q3.^2)

            ccc_integrand = cubic_color_corellator(s01,s02,q1,q2,q3,x6)
            total += s * ccc_integrand / q32
        end
        # Denominator
        q22 = sum(q2 .^ 2)
        total /= q22
        f[1] = real(total * dq2 * dϕ)
    end
    integral, err = cuhre(integrand, 8, 1, atol=1e-12, rtol=1e-10);
    # Prefactors 
    prf = - 2π^3 * alpha_s^3 * dabc2 / Nc
    # We return the result up to a factor of k^2
    result = prf * integral[1]
    return result
end

"""
    gluon_sivers(k)

Compute the gluon Sivers function for momentum transfer k
# Arguments
- `k::Vector{<:Real}`: Momentum transfer
    - 2d real vector

# Returns
Value of the gluon Sivers function at k

# To do
Prefactors need to be checked, kinematics should be ok.
"""
function gluon_sivers(k::Vector{<:Real})
    # Spin flip
    s01 = 1
    s02 = -1
    # Zero momentum transfer
    Δ = [0,0]

    odderon_dist = odderon_distribution(s01,s02,k,Δ)
    # 1 / k^2 cancelled with Odderon distribution
    prf = - mN * Nc / (8π^3 * alpha_s)
    result = prf * odderon_dist
    return result
end

end # module