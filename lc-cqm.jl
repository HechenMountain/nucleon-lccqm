module LC_CQM

# Contains wavefunction related code for the 
# light-cone constituent quark model (LC-CQM)
# of https://arxiv.org/pdf/0806.2298, 
# see also references [11-18] therein.

# Parameters are defined in parameters.jl
# Helpers are defined in helpers.jl

# ======================
# Imports
# ======================

BasePath = @__DIR__
HelpersPath = joinpath(BasePath, "helpers.jl")
ParametersPath = joinpath(BasePath, "parameters.jl")

# Integration
using Cuba

# Helpers, coordinate transformations, etc.
include(HelpersPath)
import .Helpers as hp
# Parameters and SU(Nc) algebra set in parameters.jl
include(ParametersPath)
import .Parameters: params

# ======================
# Exports
# ======================

export spin_wavefunction,
       momentum_space_wavefunction,
       baryon_wavefunction,
       compute_wavefunction,
       spin_sum,
       normalize_wavefunction

# All possible proton and constituent quark spin configurations
# Eq.(31) to (38) in the reference above, or Eq.(22) and (23) in the draft
# Maps spin tuple with externally assigned 
# kinematical parameters to expression
# Spin flip obtained by flippng sign of k_{L,R} and overall sign
const SPIN_MAP = Dict{NTuple{4, Int}, Function}(
    # s0 = +1
    (+1, +1, +1, -1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) ->  2a1*a2*a3 + a1*k2L*k3R + k1L*a2*k3R,
    (+1, +1, -1, +1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> -a1*a2*a3 + k1L*k2R*a3 - 2a1*k2R*k3L,
    (+1, -1, +1, +1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> -a1*a2*a3 + k1R*k2L*a3 - 2k1R*a2*k3L,     
    (+1, +1, -1, -1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) ->  a1*a2*k3R - 2a1*k2R*a3 - k1L*k2R*k3R,    # odd k
    (+1, -1, +1, -1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) ->  a1*a2*k3R - 2k1R*a2*a3 - k1R*k2L*k3R,    # odd k
    (+1, -1, -1, +1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) ->  a1*k2R*a3 + k1R*a2*a3 + 2k1R*k2R*k3L,    # odd k
    (+1, +1, +1, +1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> -a1*k2L*a3 - k1L*a2*a3 + 2a1*a2*k3L,      # odd k
    (+1, -1, -1, -1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> -a1*k2R*k3R - k1R*a2*k3R + 2k1R*k2R*a3,
    # s0 = -1
    (-1, -1, -1, +1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> - (2a1*a2*a3 + a1*k2R*k3L + k1R*a2*k3L),
    (-1, -1, +1, -1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> - (-a1*a2*a3 + k1R*k2L*a3 - 2a1*k2L*k3R),
    (-1, +1, -1, -1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> - (-a1*a2*a3 + k1L*k2R*a3 - 2k1L*a2*k3R),
    (-1, -1, +1, +1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> - (-a1*a2*k3L + 2a1*k2L*a3 + k1R*k2L*k3L),
    (-1, +1, -1, +1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> - (-a1*a2*k3L + 2k1L*a2*a3 + k1L*k2R*k3L),
    (-1, +1, +1, -1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> - (-a1*k2L*a3 - k1L*a2*a3 - 2k1L*k2L*k3R),
    (-1, -1, -1, -1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> - (+a1*k2R*a3 + k1R*a2*a3 - 2a1*a2*k3R),
    (-1, +1, +1, +1) => (a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R) -> - (-a1*k2L*k3L - k1L*a2*k3L + 2k1L*k2L*a3),
);

# ======================
# Wavefunctions
# ======================


"""
    spin_wavefunction(s0::Integer,
                      s1::Integer,s2::Integer,s3::Integer,
                      x1::Real, x2::Real, x3::Real,
                      k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real})

Spin dependence of baryon wavefunction obtained from Clebsch-Gordan coefficients.
Index 0 refers to proton, 1-3 are valence quark indices

#  Arguments
- `s0`: Spin of the proton (must be either +1 or -1)
- `s1, s2, s3`: Spins of the valence quarks (must be either +1 or -1)
- `x1, x2, x3`: Parton x values satisfying x1 + x2 + x3 = 1
- `k1, k2, k3`: Transverse momenta (2D cartesian vectors)

# Returns
- `wf::ComplexF64`: Value of spin wavefunction

# Notes
- Momenta must be in cartesian coordinates
"""
function spin_wavefunction(s0::Integer,
                           s1::Integer, s2::Integer, s3::Integer,
                           x1::Real, x2::Real, x3::Real,
                           k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real})
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

    k1L, k1R = complex(k1[1],-k1[2]) , complex(k1[1],k1[2]) 
    k2L, k2R = complex(k2[1],-k2[2]) , complex(k2[1],k2[2]) 
    k3L, k3R = complex(k3[1],-k3[2]) , complex(k3[1],k3[2]) 

    n1, n2, n3 = k12 + a1^2, k22 + a2^2, k32 + a3^2
    norm = 1 / sqrt(6*n1*n2*n3)
    wf = norm * SPIN_MAP[(s0,s1,s2,s3)](a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R)
    return wf
end

"""
    momentum_space_wavefunction(x1::Real, x2::Real, x3::Real,
                                k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real})

Momentum dependence of baryon wavefunction

# Arguments
- `x1, x2, x3`: Parton x values satisfying x1 + x2 + x3 = 1
- `k1, k2, k3`: Transverse momenta (2D cartesian vectors)

# Returns
Value of momentum space wavefunction

# Notes
- Momenta should be cartesian.
- Wavefunction type (exp or pow) is set in parameters.jl
"""
function momentum_space_wavefunction(x1::Real, x2::Real, x3::Real,
                                     k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real})
    # Parameters
    mq = params.mq
    β = params.β

    mq2 = mq^2
    m02 = (sum(k1.^2) + mq2) / x1 + (sum(k2.^2) + mq2) / x2 + (sum(k3.^2) + mq2) / x3
    wf_type = params.wf_type
    if wf_type == "pow"
        p = params.p
        ms_wf = (1 + m02 / β^2)^(-p)
    elseif wf_type == "exp"
        ms_wf = exp(-m02 / (2 * β^2))
    else
        error("Invalid wavefunction type: $wf_type. Must be either \"exp\" or \"pow\".")
    end
    return ms_wf
end

"""
    baryon_wavefunction(s0::Integer,
                        s1::Integer,s2::Integer,s3::Integer,
                        x1::Real, x2::Real, x3::Real,
                        k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real})
Compute product of spinor and momentum space wave function.

# Arguments
- `s0`: Spin of the baryon (must be either +1 or -1)
- `s1, s2, s3`: Spins of the valence quarks (must be either +1 or -1)
- `x1, x2, x3`: Parton x values satisfying x1 + x2 + x3 = 1
- `k1, k2, k3`: Transverse momenta (2D cartesian vectors)

# Returns
- `wf::ComplexF64`: Value of baryon wavefunction

# Notes
- 0 refers to baryon, 1-3 are valence quark indices
- Normalization carried out in f_form_factor and odderon_distribution later on
- Momenta should be cartesian.
"""
function baryon_wavefunction(s0::Integer,
                             s1::Integer,s2::Integer,s3::Integer,
                             x1::Real, x2::Real, x3::Real,
                             k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real})
    ms_wf =  momentum_space_wavefunction(x1, x2, x3, k1, k2, k3)
    spin_wf = spin_wavefunction(s0, s1, s2, s3, x1, x2, x3, k1, k2, k3)
    wf =  ms_wf * spin_wf / sqrt(3)
    return wf
end

"""
    compute_wavefunction(s::Integer,
                         x1::Real, x2::Real, x3::Real
                         k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real})

Precompute wavefunction and write index combinatioons to array.

# Arguments
- `s0`: Spin of the proton (must be either +1 or -1)
- `x1, x2, x3`: Parton x values satisfying x1 + x2 + x3 = 1
- `k1, k2, k3`: Transverse momenta (2D cartesian vectors)

# Returns
-`wf::Array{ComplexF64, 3}`: Value of norm of baryon wavefunction. To be set in parameters.jl

# Notes
- The indices 1 (-1) of the valence quarks are mapped to array indices 1 (2) 
  such that the wavefunction is accessed as e.g. wf[1,2,1]
"""
function compute_wavefunction(s::Integer,
                              x1::Real, x2::Real, x3::Real,
                              k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real})
    # initialize array with 2^3 spin configurations
    wf = Array{ComplexF64}(undef, 2, 2, 2)
    for s1 in (-1,1), s2 in (-1,1), s3 in (-1,1)
        i1, i2, i3 = hp.spin_index(s1), hp.spin_index(s2), hp.spin_index(s3)
        wf[i1,i2,i3] = baryon_wavefunction(s, s1, s2, s3, x1, x2, x3, k1, k2, k3)
    end
    return wf
end

"""
    spin_sum(wf1::Array{ComplexF64, 3},wf2::Array{ComplexF64, 3})

Carries out the spin sum over 2 wavefunctions.

# Arguments
- `wf1,wf2::Array{ComplexF64, 3}`: Baryon wavefunctions

# Returns
- `total::ComplexF64`: Value of ∑conj(wf2) * wf1

# Notes
- The wavefunctions wf1, wf2 passed to this object need to be in array form
  as generated by compute_wavefunction.
"""
# Wrapper for spin sum
function spin_sum(wf1::Array{ComplexF64, 3},wf2::Array{ComplexF64, 3})
    total = complex(0,0)
    # Sum over spins
    for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
        i1, i2, i3 = hp.spin_index(s1), hp.spin_index(s2), hp.spin_index(s3)
        total += conj(wf2[i1, i2, i3]) * wf1[i1, i2, i3]
    end
    return total
end

"""
    normalize_wavefunction()

Normalize the baryon wave function with parameters defined in parameters.jl

# Returns
- `norm::Float64`: Value of norm of baryon wavefunction and errors. To be set in parameters.jl afterwards
- `err::Vector{Float64}`: Array [err_re, err_im] containing error estimates for real and imaginary parts

# Notes
- Cuba samples are regulated to avoid endpoint singularities
- Summation is done inside the integrand such that cuhre is only called once.
"""
function normalize_wavefunction()
    function integrand(x,f)
        x = hp.regulate_cuba(x)
        (x1, x2, x3), dx = hp.cuba_to_parton_x(x[1:2])
        r1, ϕ1, d2k1 = hp.cuba_to_polar(x[3:4])
        r2, ϕ2, d2k2 = hp.cuba_to_polar(x[5:6])
        
        # Reconstruct cartesian momenta from polar coordinates
        k1 = hp.polar_to_cartesian([r1, ϕ1])
        k2 = hp.polar_to_cartesian([r2, ϕ2])
        k3 = - (k1 + k2)  # Enforce transverse momentum conservation

        # Jacobian
        d4k = d2k1 * d2k2

        # Precompute momentum space wavefunction
        ms_wf = momentum_space_wavefunction(x1, x2, x3, k1, k2, k3)
        # Sum over spin contributions
        total = 0.0
        for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
            spin_wf = spin_wavefunction(1, s1, s2, s3, x1, x2, x3, k1, k2, k3) 
            wf = ms_wf * spin_wf
            # total += abs2(wf)
            total += conj(wf) * wf
        end
        result = total * d4k * dx
        f[1] =  real(result)
        f[2] =  imag(result)
    end
    integral, err, prob, neval, fail, nregions = cuhre(integrand, 6, 2; maxevals=10_000_000) 
    # integral, err = vegas(integrand, 6, 2; maxevals=10_000_000)
    # Multiply with prefactors from integration
    prf = 1 / (4π)^2 / (2π)^4 
    result = prf * complex(integral[1], integral[2])
    err .*= prf
    # Warn if imaginary part is large
    if imag(result) > 1e-6
        @warn "Large imaginary part in wavefunction normalization: $(imag(result))"
    end
    # Return norm
    norm = 1 / sqrt(real(result))
    return norm, err, prob, neval, fail, nregions
end

# ======================
end # module LC_CQM