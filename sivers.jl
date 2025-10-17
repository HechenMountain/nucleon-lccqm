module Sivers

using Base: pi
# Integration
using Cuba

BasePath = joinpath(homedir(),"jupyter","julia","sivers") 
ParametersPath = joinpath(BasePath,"parameters.jl")
HelpersPath = joinpath(BasePath,"helpers.jl")

# Parameters handled separately
include(ParametersPath)
using .parameters: params

# Helpers, coordinate transformations, etc.
include(HelpersPath)
import .helpers 
# Shorthand
hp = helpers

###################

export  hp, f1_form_factor, f2_form_factor,
        normalize_wavefunction,
        regulator_scan, f_form_factor,
        cubic_color_correlator, odderon_distribution,
        compute_wavefunction, spin_sum, spin_wavefunction,
        baryon_wavefunction, write_f2_form_factor_to_csv,
        f1_form_factor_table_csv

# export  f1_form_factor, f2_form_factor,
#         cubic_color_correlator, odderon_distribution, gluon_sivers,
#         compute_wavefunction, spin_sum, normalize_wavefunction,
#         hp.SPIN_MAP,hp.kronecker_delta, hp
#         momentum_space_wavefunction, spin_wavefunction,
#         baryon_wavefunction,compute_wavefunction,
#         write_odderon_distribution_to_csv, regulator_scan

# Parameters and SU(Nc) algebra set in parameters.jl
alpha_s = params.alpha_s ;
Nc = params.Nc ;
mN = params.mN ;
dabc2 = (Nc^2 - 4) * (Nc^2 - 1) / Nc ;

#####################
### Wavefunctions ###
#####################

"""
    spin_wavefunction(s0::Integer,
                      s1::Integer,s2::Integer,s3::Integer,
                      k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real}, 
                      x1::Real, x2::Real, x3::Real)

Spin dependence of baryon wavefunction obtained from Clebsch-Gordan coefficients.
0 refers to proton, 1-3 are valence quark indices
### Input

- `s0, s1, s2, s3`  -- Spin indices of proton and valence quarks. Either +1 or -1
- `k1, k2, k3`      -- Momenta of valence quarks
                       - 2d real vectors
- `x1, x2, x3`      -- Parton x of valence quarks
                       - Must be in [0,1]

### Output

Value of spinor wavefunction

### Notes

Momenta should be cartesian.
"""
function spin_wavefunction(s0::Integer,
                           s1::Integer, s2::Integer, s3::Integer,
                           k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real}, 
                           x1::Real, x2::Real, x3::Real)
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
    wf = norm * hp.SPIN_MAP[(s0,s1,s2,s3)](a1,a2,a3,k1L,k1R,k2L,k2R,k3L,k3R)
    return wf
end

"""
    momentum_space_wavefunction(k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real},
                                x1::Real, x2::Real, x3::Real)

Momentum dependence of baryon wavefunction

### Input

- `k1, k2, k3 -- Momenta of valence quarks
                 - 2d real vectors
- `x1, x2, x3 -- Parton x of valence quarks
                 - Must be in [0,1]

### Output

Value of momentum space wavefunction

### Notes

Momenta should be cartesian.
"""
function momentum_space_wavefunction(k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real},
                                     x1::Real, x2::Real, x3::Real)
    # Parameters
    mq = params.mq
    β = params.β

    mq2 = mq^2
    m02 = (sum(k1.^2) + mq2) / x1 + (sum(k2.^2) + mq2) / x2 + (sum(k3.^2) + mq2) / x3
    return exp(-m02 / (2 * β^2))
end

"""
    baryon_wavefunction(s0::Integer,
                        s1::Integer,s2::Integer,s3::Integer,
                        k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real}, 
                        x1::Real, x2::Real, x3::Real)
Compute product of spinor and momentum space wave function.

### Input

- `s0, s1, s2, s3 -- Spin indices of proton and valence quarks. 
                     - Must be either +1 or -1
- `k1, k2, k3`    -- Momenta of valence quarks
                     - 2d real vectors
- `x1, x2, x3` -- Parton x of valence quarks
                  - Must be in [0,1]

### Output

Value of baryon wavefunction

### Notes

- 0 refers to proton, 1-3 are valence quark indices
- Normalization carried out in f_form_factor and odderon_distribution later on
- Momenta should be cartesian.
"""
function baryon_wavefunction(s0::Integer,
                             s1::Integer, s2::Integer, s3::Integer,
                             k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real}, 
                             x1::Real, x2::Real, x3::Real)
    ms_wf =  momentum_space_wavefunction(k1, k2, k3, x1, x2, x3)
    spin_wf = spin_wavefunction(s0, s1, s2, s3, k1, k2, k3, x1, x2, x3)
    wf =  ms_wf * spin_wf / sqrt(3)
    return wf
end

"""
    compute_wavefunction(s::Integer,
                         k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real},
                         x1::Real, x2::Real, x3::Real)

Precompute wavefunction and write index combinatioons to array.

### Input

- `s`          -- Spin index of proton.
                  - Must be either +1 or -1
- `k1, k2, k3` -- Momenta of valence quarks
                  - 2d real vectors
- `x1, x2, x3` -- Parton x of valence quarks
                  - Must be in [0,1]

### Output

Value of norm of baryon wavefunction. To be set in parameters.jl

### Notes

The indices 1 (-1) of the valence quarks are mapped to 1 (2) 
such that the wavefunction is accessed as e.g. wf[1,2,1]
"""
function compute_wavefunction(s::Integer,
                              k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real},
                              x1::Real, x2::Real, x3::Real)
    # initialize array with 2^3 spin configurations
    wf = Array{ComplexF64}(undef, 2, 2, 2)
    for s1 in (-1,1), s2 in (-1,1), s3 in (-1,1)
        i1, i2, i3 = hp.spin_index(s1), hp.spin_index(s2), hp.spin_index(s3)
        wf[i1,i2,i3] = baryon_wavefunction(s, s1, s2, s3, k1, k2, k3, x1, x2, x3)
    end
    return wf
end

"""
    spin_sum(wf1::Array{ComplexF64, 3},wf2::Array{ComplexF64, 3})

Carries out the spin sum over 2 wavefunctions.

### Input
- `wf1,wf2` -- Baryon wavefunctions
               - Must be array with 2^3 entries

### Output
Value of ∑conj(wf2) * wf1

### Notes

The wavefunctions wf1, wf2 passed to this object need to be in array form
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

### Output
Value of norm of baryon wavefunction and errors. To be set in parameters.jl

### Notes
- Cuba samples are regulate to avoid endpoint singularities
- Summation is done inside the integrand such that cuhre is only called once.
"""
function normalize_wavefunction()
    function integrand(x,f)
        x = hp.regulate_cuba(x)
        (x1, x2, x3), dx = hp.cuba_to_parton_x(x[1:2])
        r1, ϕ1, d2k1 = hp.cuba_to_polar(x[3:4])
        r2, ϕ2, d2k2 = hp.cuba_to_polar(x[5:6])
        
        # Reconstruct cartesian momenta from polar coordinates
        k1 = [r1 * cos(ϕ1), r1 * sin(ϕ1)]
        k2 = [r2 * cos(ϕ2), r2 * sin(ϕ2)]
        k3 = - (k1 + k2)  # Enforce transverse momentum conservation

        # Jacobian
        d4k = d2k1 * d2k2

        # Precompute momentum space wavefunction
        ms_wf = momentum_space_wavefunction(k1, k2, k3, x1, x2, x3)
        # Sum over spin contributions
        total = 0.0
        for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
            spin_wf = spin_wavefunction(1, s1, s2, s3, k1, k2, k3, x1, x2, x3) 
            wf = ms_wf * spin_wf
            # total += abs2(wf)
            total += conj(wf) * wf
        end
        result = total * d4k * dx
        f[1] =  real(result)
        f[2] =  imag(result)
    end
    integral, err = cuhre(integrand, 6, 2; maxevals=10_000_000) 
    # integral, err = vegas(integrand, 6, 2; maxevals=10_000_000)
    # Multiply with prefactors from integration
    result = 1 / (4π)^2 / (2π)^4 * complex(integral[1], integral[2])
    # Return norm
    norm = 1 / sqrt(result)
    return norm, err
end

####################
### Form factors ###
####################

"""
    f_form_factor(s01::Integer, s02::Integer, Δ::Vector{<:Real})

Compute the F-type form factor needed to generate F1 and F2

### Input

- `s01, s02` -- Spins of the external protons
                - Must be either +1 or -1
- `Δ`        -- Momentum transfer
                - 2d real vector

### Output

Value of the F-type form factor for a given momentum transfer

### Notes

- norm is set in parameters.jl and can be obtained from normalize_wavefunction(). 
- Spin sums etc. are performed inside integrand, then cuhre is used once.
- Momenta need to be cartesian
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
        k1 = [r1 * cos(ϕ1), r1 * sin(ϕ1)]
        k2 = [r2 * cos(ϕ2), r2 * sin(ϕ2)]
        k3 = - (k1 + k2)  # Enforce transverse momentum conservation

        # Jacobian
        d4k = d2k1 * d2k2

        total = 0
        wf1 = compute_wavefunction(s01, k1, k2, k3, x1, x2, x3)
        # Sum over charge contributions
        for (i,q) in enumerate(charges)
            k1prime = k1 - x1 * Δ + hp.kronecker_delta(i,1) * Δ
            k2prime = k2 - x2 * Δ + hp.kronecker_delta(i,2) * Δ
            k3prime = k3 - x3 * Δ + hp.kronecker_delta(i,3) * Δ
            wf2 = compute_wavefunction(s02, k1prime, k2prime, k3prime, x1, x2, x3)
            # Sum over spin contributions
            total += q * spin_sum(wf1, wf2)
        end
        res = total * d4k * d2x
        f[1] = real(res)
        f[2] = imag(res)
    end
    # Call cuhre with ncomp=2 to track real and imaginary parts separately
    integral, err = cuhre(integrand, 6, 2; maxevals=10_000_000)
    # integral, err = vegas(integrand, 6, 2; maxevals=10_000_000)
    # Reconstruct complex result and
    # multiply with prefactors from integration
    norm = params.norm
    prf = 3 / (4π)^2 / (2π)^4 * norm^2
    result = prf * complex(integral[1], integral[2])
    err .*= prf
    return result, err
end

"""
    f1_form_factor(Δ::Vector{<:Real})

Compute the F1 form factor

### Input

- `Δ` -- Momentum transfer
         - 2d real vector

### Output

Value of the F1 form factor for a given momentum transfer

### Notes

- norm is set in parameters.jl and can be obtained from normalize_wavefunction(s0)
- Spin sums etc. are performed inside integrand, then cuhre is used once.
- Momenta need to be cartesian
"""
function f1_form_factor(Δ::Vector{<:Real})
    result, err = f_form_factor(1, 1, Δ)
    return result, err
end

"""
    f2_form_factor(Δ::Vector{<:Real})

Compute the F2 form factor

### Input

- `Δ` -- Momentum transfer
         - 2d real vector

### Output

Value of the F2 form factor for a given momentum transfer

### Notes

- norm is set in parameters.jl and can be obtained from normalize_wavefunction(s0)
- Spin sums etc. are performed inside integrand, then cuhre is used once.
- Momenta need to be cartesian
"""
function f2_form_factor(Δ::Vector{<:Real})
    ΔL, ΔR = complex(Δ[1], -Δ[2]) , complex(Δ[1], Δ[2]) 
    Δ2 = sum(Δ.^2)
    # Notation in notes reversed: Lambda', Lambda = s02, s01
    fdu, err_du = f_form_factor(1, -1, Δ)
    fud, err_ud = f_form_factor(-1, 1, Δ)

    result = mN^2 / Δ2 * (ΔL / mN * fdu - ΔR / mN * fud)
    # imaginary part cancels
    err_re = mN / sqrt(Δ2) * sqrt(err_du[1]^2 + err_ud[1]^2)
    err_im = mN / sqrt(Δ2) * sqrt(err_du[2]^2 + err_ud[2]^2)
    return result, [err_re, err_im]
end

###########################
###     Distributions   ###
###########################

"""
    cubic_color_correlator(s01::Integer,s02::Integer,
                           x1::Real, x2::Real, x3::Real,
                           q1::Vector{<:Real}, q2::Vector{<:Real}, q3::Vector{<:Real},
                           k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real})

Compute the unintegrated cubic color correlator by summing one-, two-, and three-body contributions.

### Input

- `s01, s02`    -- Spins of the external protons
                   - Must be either +1 or -1
- `x1, x2, x3`  -- Parton-x values
                   - (1 - x1 -x2 - x3) = 0
- 'q1, q2, q3'  -- Eikonal momenta
                   - 2d real vectors
- 'k1, k2, k3'  -- Transverse momenta
                   - 2d real vectors

### Output

- Value of the cubic color correlator for the given spin configuration and kinematics

### Notes

This is G_3ΛΛ' stripped of the integrals and factor of  1 / (4π)^2 / (2π)^2  in the draft
"""
function cubic_color_correlator(s01::Integer,s02::Integer,
                                x1::Real, x2::Real, x3::Real,
                                q1::Vector{<:Real}, q2::Vector{<:Real}, q3::Vector{<:Real},
                                k1::Vector{<:Real}, k2::Vector{<:Real}, k3::Vector{<:Real})
    function one_body_kin(i, j123, q1, q2, q3)
        # Momentum inflow [q1 + q2 + q3] at j123
        # i is k_prime (quark) index, j123 quark line with 
        # momentum inflow q1 + q2 + q3 from gluon.
        delta_kiprime =  hp.kronecker_delta(i, j123) * (q1 + q2 + q3)
        return delta_kiprime
    end
    function two_body_kin(i, j12, j3, l, q1, q2, q3)
        # Momentum inflow [q1 + q2,j12] [q3,j3] at j12 and j3
        # i is k_prime (quark) index, j12, j3 quark line with
        # momentum inflow q1 + q2 and q3 from gluon.
        # Addtional terms from permutations, so in total 3 contributions
        # which we distinguish by l
        if l == 1 # [q2 + q3,j12] [q1,j3]
            delta_kiprime =  hp.kronecker_delta(i, j12) * (q2 + q3) + hp.kronecker_delta(i, j3) * q1
        elseif l == 2 # [q1 + q3,j12] [q2,j3]
            delta_kiprime =  hp.kronecker_delta(i, j12) * (q1 + q3) + hp.kronecker_delta(i, j3) * q2
        elseif l == 3 # [q1 + q2,j12] [q3,j3]
            delta_kiprime =  hp.kronecker_delta(i, j12) * (q1 + q2) + hp.kronecker_delta(i, j3) * q3
        end
        return delta_kiprime
    end
    function three_body_kin(i, j1, j2, j3, q1, q2, q3)
        # Momentum inflow [q1,j1] [q2,j2] [q3,j3]
        # i is k_prime (quark) index, j1, j2, j3 are gluons with momenta
        # q1, q2 and q3, respectively, attached to quark lines.
        delta_kiprime = hp.kronecker_delta(i, j1) * q1 + hp.kronecker_delta(i, j2) * q2 + hp.kronecker_delta(i, j3) * q3
        return delta_kiprime
    end
    # Cubic color correlator without Jacobian
    # Precompute incoming baryon wavefunction
    wf1 = compute_wavefunction(s01, k1, k2, k3, x1, x2, x3)

    # Constant parts
    Δ = - (q1 + q2 + q3)
    k1prime0, k2prime0, k3prime0 = k1 - x1 * Δ, k2 - x2 * Δ, k3 - x3 * Δ
    # Sum over one-body, two-body and three-body kinematics
    total = complex(0,0)
    # One-body
    for j123 in 1:3
        k1prime = k1prime0 - one_body_kin(1, j123, q1, q2, q3)
        k2prime = k2prime0 - one_body_kin(2, j123, q1, q2, q3)
        k3prime = k3prime0 - one_body_kin(3, j123, q1, q2, q3)
        wf2 = compute_wavefunction(s02,k1prime,k2prime,k3prime,x1,x2,x3)
        # Perform spin sum
        total += spin_sum(wf1, wf2)
    end
    # Two-body
    # Addtional terms from permutations, so we have an extra sum over k
    for l in 1:3, j12 in 1:3, j3 in 1:3
        if j12 == j3
            continue
        end
        k1prime = k1prime0 - two_body_kin(1, j12, j3, l, q1, q2, q3)
        k2prime = k2prime0 - two_body_kin(2, j12, j3, l, q1, q2, q3)
        k3prime = k3prime0 - two_body_kin(3, j12, j3, l, q1, q2, q3)
        wf2 = compute_wavefunction(s02, k1prime, k2prime, k3prime, x1, x2, x3)
        # Perform spin sum
        total += spin_sum(wf1, wf2)
    end
    # Three-body
    for j1 in 1:3, j2 in 1:3, j3 in 1:3
        if j1 == j2 || j1 == j3 || j2 == j3
            continue
        end
        k1prime = k1prime0 - three_body_kin(1, j1, j2, j3, q1, q2, q3)
        k2prime = k2prime0 - three_body_kin(2, j1, j2, j3, q1, q2, q3)
        k3prime = k3prime0 - three_body_kin(3, j1, j2, j3, q1, q2, q3)
        wf2 = compute_wavefunction(s02, k1prime, k2prime, k3prime, x1, x2, x3)
        # Perform spin sum
        total += spin_sum(wf1, wf2)
    end
    return total
end

"""
    odderon_distribution(s01::Integer,s02::Integer,
                         k::Vector{<:Real},Δ::Vector{<:Real};
                         [mu]::Real=0.01,[solver]::String="cuhre")

Compute the Odderon distribution O * k^2 for momentum transfer k and Δ

### Input

- `s01, s02` -- Spins of the external protons
                - Must be either +1 or -1
- `k`        -- Transverse momentum transfer
                - 2d real vector
- `Δ`        -- Total momentum transfer
                - 2d real vector
- `mu`       -- (optional, default: `0.01`) Regulator for integrand
- `solver`   -- (optional, default: `"cuhre"`) Integration strategy
                - Either "cuhre", "vegas", "divonne", "suave"

### Output

Value of the the Odderon distribution times k^2 at k and Δ.

### Notes
- For now this expression is only valid for Δ = [0,0].
  Corresponds to OΛΛ'(k,Δ) in the draft.
- We drop the 1 / k^2 and add it later in the definition of the sivers function. 
- Supplied momenta should be cartesian.
- norm is set in parameters.jl and can be obtained from normalize_wavefunction(). 
"""
function odderon_distribution(s01::Integer, s02::Integer,
                              k::Vector{<:Real}, Δ::Vector{<:Real};
                              mu::Real=0.01, solver::String="cuhre")
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
        k1 = [r1 * cos(ϕ1), r1 * sin(ϕ1)]
        k2 = [r2 * cos(ϕ2), r2 * sin(ϕ2)]
        k3 = - (k1 + k2)  # Enforce transverse momentum conservation
        d4k = d2k1 * d2k2

        q2 = [r3 * cos(ϕ3), r3 * sin(ϕ3)]
        # Jacobian
        d8x = d2x * d4k * d2q2 # 2 + 4 + 2 = 8d integral

        total = 0
        for s in (+1,-1)
            # Flip momenta to project out Sivers function
            q1, q2, q3 = s * k, s * q2, - s * (k + q2)
            ccc = cubic_color_correlator(s01, s02, x1, x2, x3, q1, q2, q3, k1, k2, k3)
            total += s * ccc
        end
        # Regenerate initial q2
        q2 = [r3 * cos(ϕ3), r3 * sin(ϕ3)]
        q3 = k + q2
        q22 = sum(q2.^2)
        q32 = sum(q3.^2)
        # Add regulator
        mu2 = mu^2
        q22 += mu2
        q32 += mu2
        # Same denominator
        # for both terms once momenta
        # have been flipped
        den = q22 * q32  
        total *=  d8x / den

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
    odderon_distribution_v2(k,Δ)

Compute the Odderon distribution O * k^2 for momentum transfer k and Δ

### Input
- `s01, s02` -- Spin of the incoming hadron
- `k`        -- Transverse momentum transfer
                - 2d real vector
- `Δ`        -- Total momentum transfer
                - 2d real vector
- `mu`       -- Regulator for integrand
- `solver`   -- Integration strategy
                - Either "cuhre", "vegas", "divonne", "suave"

### Output

Value of the the Odderon distribution times k^2 at k and Δ. Does not utilize
the symmetry of the integrand when k_i -> - k_i

### Notes

- For now this is expression is only valid for Δ = [0,0].
  Corresponds to OΛΛ'(k,Δ) in the draft.
- We drop the 1 / k^2 since it cancels with the corresponding
  factor in the definition of the sivers function. 
- Supplied momenta should be cartesian.
"""
function odderon_distribution_v2(s01::Integer,s02::Integer,
                                 k::Vector{<:Real},Δ::Vector{<:Real};
                                 mu::Real=0.01)
    if !iszero(Δ)
        throw(ArgumentError("Implementation currently only for vanishing Δ."))
    end

    function integrand(x,f)
        # Transform [0,1]^8 cuba samples to physical variables
        # Parton-x
        (x1, x2, x3), d2x = hp.cuba_to_parton_x(x[1:2])
        # Momenta
        r1, ϕ1, d2k1 = hp.cuba_to_polar(x[3:4])    # k1
        r2, ϕ2, d2k2 = hp.cuba_to_polar(x[5:6])    # k2
        r3, ϕ3, d2q2 = hp.cuba_to_polar(x[7:8])    # q2
        
        # Reconstruct cartesian momenta from polar coordinates
        k1 = [r1 * cos(ϕ1), r1 * sin(ϕ1)]
        k2 = [r2 * cos(ϕ2), r2 * sin(ϕ2)]
        k3 = - (k1 + k2)  # Enforce transverse momentum conservation
        d4k = d2k1 * d2k2

        q2 = [r3 * cos(ϕ3), r3 * sin(ϕ3)]
        # Jacobian
        d8x = d2x * d4k * d2q2 # 2 + 4 + 2 = 8d integral

        # Regulator
        mu2 = mu^2

        total = complex(0,0)
        for s in (+1,-1)
            # Flip momenta to project out Sivers function
            q1, q3 = s * k, - s * k - q2
            q32 = sum(q3.^2)
            # Add regulator
            q32 += mu2
            # k1, k2, k3 = s * k1, s * k2, s * k3  
            ccc = cubic_color_correlator(s01,s02,x1,x2,x3,q1,q2,q3,k1,k2,k3)
            total += s * ccc / q32
        end
        q22 = sum(q2.^2)
        # Add regulator
        q22 += mu2
        total *=  d8x / q22

        f[1] = real(total)
        f[2] = imag(total)
    end
    integral, err = cuhre(integrand, 8, 2; maxevals=10_000_000);
    # Prefactors 
    # prf = - 2π^3 * alpha_s^3 * dabc2 / Nc
    # For now, factor of g^6 = 1, for simplicity
    prf = - dabc2 / Nc / 32 
    # π factors from integration
    prf /= (4π)^2 * (2π)^6
    # We return the result up to a factor of k^2
    result = prf * complex(integral[1], integral[2])
    err .*= abs(prf)
    return result, err
end

"""
    gluon_sivers(k::Real;mu::Real=0.01,solver::String="cuhre")

Compute the gluon Sivers function for momentum transfer k

### Input

- `k` -- Transverse momentum transfer
   - 2d real vector
- `Δ` -- Total momentum transfer
   - 2d real vector
- `mu` -- Regulator for integrand
- `solver` -- (optional, default =`"cuhre`) Integration strategy
    - Either "cuhre", "vegas", "divonne", "suave"

### Output

- Value of the gluon Sivers function at k

### Notes
Assumes that 2d k vector is [kx, 0]. For k not along x,
one would need to compute both s01, s02 = 1, -1 and -1, 1 and sum
like in f2_form_factor
"""
function gluon_sivers(k::Real; mu::Real=0.01, solver::String="cuhre")
    # Spin flip
    s01 = 1
    s02 = -1
    # Zero momentum transfer
    Δ = [0,0]

    odderon_dist, = odderon_distribution(s01, s02, [k,0], Δ; mu=mu, solver=solver)
    # 1 / k^2 partially cancels with Odderon distribution
    k_perp = k[1]
    # Regulate singularity
    if k_perp == 0
        k_perp = 1e-12
    end
    prf = - 8 * mN * Nc / π * alpha_s^2 / k[1]
    result = prf * odderon_dist
    return result
end

#################
### Write out ###
#################

"""
    write_odderon_distribution_to_csv(mu::Real, [solver]::String="cuhre")

Write result of odderon_distribution for |k| in [0,2] GeV

### Input

- `mu` -- Regulator for integration
- `solver` -- (optional, default =`"cuhre`) Integration strategy
  - Either "cuhre", "vegas", "divonne", "suave"

### Notes

- Run over ssh using e.g.
  nohup julia -e 'include("sivers.jl"); 
  Sivers.write_odderon_distribution_to_csv(mu)' > log.txt 2>&1 &
"""
function write_odderon_distribution_to_csv(mu::Real, solver::String="cuhre")
    filename = "output_$(solver)_$(mu).csv"
    open(filename, "w") do io 
        println(io, "k,val_re,val_im,err_re,err_im,prob_re,prob_im,neval,fail,nregions")  # header
        for k in 0:0.01:1.25
            # val = odderon_distribution(1,-1,[k,0],[0,0],mu)
            # println(io, "$(k),$(real(val)),$(imag(val))")
            integral, err, prob, neval, fail, nregions  = odderon_distribution(1, -1, [k,0], [0,0]; mu=mu, solver=solver)
            println(io, "$(k),$(integral[1]),$(integral[2]),$(err[1]),$(err[2]),$(prob[1]),$(prob[2]),$(neval),$(fail),$(nregions)")
            flush(io) 
        end
    end
end

"""
    regulator_scan(solver::String)

Write result of F2 form factor for |Δ| in [1e-6,3.3] GeV

### Input

- `solver` -- Integration strategy
    - Either "cuhre", "vegas", "divonne", "suave"

### Notes

- Run over ssh using e.g.
  nohup julia -e 'include("sivers.jl"); 
  Sivers.write_f2_form_factor_to_csv()' > log.txt 2>&1 &
"""
function regulator_scan(solver::String)
    mu_vals = [0.01, 0.02, 0.03, 0.04, 0.05]
    for mu in mu_vals
        write_odderon_distribution_to_csv(mu, solver)
    end
end

"""
    write_f1_form_factor_to_csv()

Write result of F1 form factor for |Δ| in [0,3.3] GeV

### Notes
- Run over ssh using e.g.
  nohup julia -e 'include("sivers.jl"); 
  Sivers.write_f1_form_factor_to_csv()' > log.txt 2>&1 &
"""
function write_f1_form_factor_to_csv()
    open("output_f1.csv", "w") do io
        println(io, "k,val,err_real,err_imag")  # header
        for Δ in 0.0:.125:3.3
            val, err = f1_form_factor([Δ, 0])
            println(io, "$(Δ),$(val),$(err[1]),$(err[2])")
            flush(io) 
        end
    end
end

"""
    write_f2_form_factor_to_csv()

Write result of F2 form factor for |Δ| in [1e-6,3.3] GeV

### Notes
- Run over ssh using e.g.
  nohup julia -e 'include("sivers.jl"); 
  Sivers.write_f2_form_factor_to_csv()' > log.txt 2>&1 &
"""
function write_f2_form_factor_to_csv()
    open("output_f2.csv", "w") do io
        println(io, "k,val,err_real,err_imag")  # header
        for Δ in 1e-6:.125:3.3
            val, err = f2_form_factor([Δ, 0])
            println(io, "$(Δ),$(val),$(err[1]),$(err[2])")
            flush(io) 
        end
    end
end

function write_f_form_factor_to_csv()
    open("output_f.csv", "w") do io
        println(io, "k,val,err_real,err_imag")  # header
        for Δ in 0.01:.125:3.3
            val, err = f_form_factor(1,-1,[Δ, 0])
            println(io, "$(Δ),$(val),$(err[1]),$(err[2])")
            flush(io) 
        end
    end
end


#############
#############
end # module