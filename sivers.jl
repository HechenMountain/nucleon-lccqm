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

export  SPIN_MAP,kronecker_delta,
        momentum_space_wavefunction, spin_wavefunction,
        baryon_wavefunction, f1_form_factor, f1_form_factor_table

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
    normalize_wavefunction(s0)

Normalize the baryon wave function with parameters defined in parameters.jl
# Arguments
- `s0::Integer`: Spin index of proton.
    - Must be either +1 or -1

# Returns
Value of norm of baryon wavefunction. To be set in parameters.jl

# Notes
norm is set in parameters.jl and can be obtained from normalize_wavefunction(s0).
Summation is done inside the integrand such that cuhre is only called once.
There is a commented out version in the module where individual contributions
are integrated separately and then summed. This gives a slight speedup, might be
useful in the future.
"""
function normalize_wavefunction(s0::Integer)
    function integrand(x,f)
        x1 = x[1]
        x2 = (1 - x1) * x[2]
        x3 = 1 - x1 - x2
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

        k3 = k1 - k2  # Enforce transverse momentum conservation

        # Jacobian
        dk1 = r1/(1 - x[3])^2 # r1
        dk2 = r2/(1 - x[5])^2 # r2
        dϕ = (2π)^2 # dϕ1 * dϕ2
        dk = dk1 * dk2

        ms_wf = momentum_space_wavefunction(k1, k2, k3, x1, x2, x3)
        # Sum over spin contributions
        total = 0.0
        for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
            spin_wf = spin_wavefunction(s0,s1,s2,s3,k1,k2,k3,x1,x2,x3) 
            wf = ms_wf * spin_wf
            total += abs2(wf)
        end
        f[1] =  total * dk * dϕ
    end
    integral, err = cuhre(integrand, 6, 1, atol=1e-8, rtol=1e-6);
    # Multiply with prefactors
    result = 1/(4π)/(2π)^2 * integral[1]
    # Return norm
    norm = 1/sqrt(result)
    return norm
end

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
    function integrand(x,f)
        x1 = x[1]
        x2 = (1 - x1) * x[2]
        x3 = 1 - x1 - x2
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

        k3 = k1 - k2  # Enforce transverse momentum conservation

        # Jacobian
        dk1 = r1/(1 - x[3])^2 # r1
        dk2 = r2/(1 - x[5])^2 # r2
        dϕ = (2π)^2 # dϕ1 * dϕ2
        dk = dk1 * dk2

        total = 0
        # Sum over charge contributions
        for (i,q) in enumerate(charges)
        k1prime = k1 - x1 * Δ + kronecker_delta(i,1) * Δ
        k2prime = k2 - x2 * Δ + kronecker_delta(i,2) * Δ
        k3prime = k3 - x3 * Δ + kronecker_delta(i,3) * Δ
            # Sum over spin contributions
            for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
                # Both wave functions have spin up
                wf1 = baryon_wavefunction(1,s1,s2,s3,k1,k2,k3,x1,x2,x3)
                wf2 = baryon_wavefunction(1,s1,s2,s3,k1prime,k2prime,k3prime,x1,x2,x3)
                total += q * conj(wf2) * wf1
            end
        end
        res = total * dk * dϕ
        # if abs(imag(res)) > 1e-10
        #     error("Imaginary part in integrand: imag = $(imag(res))")
        # end
        f[1] = real(res)
    end
    integral, err = cuhre(integrand, 6, 1, atol=1e-12, rtol=1e-10);
    result =  3 / (4π) / (2π)^2 * integral[1]
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

#                 k3 = k1 - k2  # Enforce transverse momentum conservation

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
               
#                 res = q * conj(wf2) * wf1 * dk * dϕ
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

end # module