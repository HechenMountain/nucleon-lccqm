module Sivers

using Base: pi
# Integration
using Cuba

BasePath = @__DIR__
ParametersPath = joinpath(BasePath,"parameters.jl")
HelpersPath = joinpath(BasePath,"helpers.jl")
LCQMPath = joinpath(BasePath,"lc-cqm.jl")

# Parameters handled separately
include(ParametersPath)
using .Parameters: params

# Light-cone constituent quark model wavefunctions
include(LCQMPath)
import .LC_CQM as wfs

# Helpers, coordinate transformations, etc.
include(HelpersPath)
import .Helpers as hp

###################

export  hp, lc_cqm, 
        f1_form_factor,
        f2_form_factor,
        f_form_factor,
        cubic_color_correlator, 
        odderon_distribution,
        regulator_scan,
        write_f1_form_factor_to_csv,
        write_f2_form_factor_to_csv,
        write_odderon_distribution_to_csv,
        write_odderon_distribution_range_to_csv

# Parameters and SU(Nc) algebra set in parameters.jl
alpha_s = params.alpha_s ;
Nc = params.Nc ;
mN = params.mN ;
dabc2 = (Nc^2 - 4) * (Nc^2 - 1) / Nc ;

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
        wf1 = wfs.compute_wavefunction(s01, k1, k2, k3, x1, x2, x3)
        # Sum over charge contributions
        for (i,q) in enumerate(charges)
            k1prime = k1 - x1 * Δ + hp.kronecker_delta(i,1) * Δ
            k2prime = k2 - x2 * Δ + hp.kronecker_delta(i,2) * Δ
            k3prime = k3 - x3 * Δ + hp.kronecker_delta(i,3) * Δ
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
    # integral, err = vegas(integrand, 6, 2; maxevals=10_000_000)
    # Reconstruct complex result and
    # multiply with prefactors from integration
    norm = params.norm
    prf = 3 / (4π)^2 / (2π)^4 * norm^2
    result = prf * complex(integral[1], integral[2])
    err .*= prf
    return result, err, prob, neval, fail, nregions
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

- norm is set in parameters.jl and can be obtained from normalize_wavefunction()
- Spin sums etc. are performed inside integrand, then cuhre is used once.
- Momenta need to be cartesian
"""
function f1_form_factor(Δ::Vector{<:Real})
    result, err, prob, neval, fail, nregions = f_form_factor(1, 1, Δ)
    return result, err, prob, neval, fail, nregions
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

- norm is set in parameters.jl and can be obtained from normalize_wavefunction()
- Spin sums etc. are performed inside integrand, then cuhre is used once.
- Momenta need to be cartesian
"""
function f2_form_factor(Δ::Vector{<:Real})
    ΔL, ΔR = complex(Δ[1], -Δ[2]) , complex(Δ[1], Δ[2]) 
    Δ2 = sum(Δ.^2)
    # Notation in notes reversed: Lambda', Lambda = s02, s01
    fdu, err_du, prob, neval, fail, nregions = f_form_factor(1, -1, Δ)
    fud, err_ud, = f_form_factor(-1, 1, Δ)

    result = mN^2 / Δ2 * (ΔL / mN * fdu - ΔR / mN * fud)
    # imaginary part cancels
    err_re = mN / sqrt(Δ2) * sqrt(err_du[1]^2 + err_ud[1]^2)
    err_im = mN / sqrt(Δ2) * sqrt(err_du[2]^2 + err_ud[2]^2)
    return result, [err_re, err_im], prob, neval, fail, nregions
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
    wf1 = wfs.compute_wavefunction(s01, k1, k2, k3, x1, x2, x3)

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
        wf2 = wfs.compute_wavefunction(s02,k1prime,k2prime,k3prime,x1,x2,x3)
        # Perform spin sum
        total += wfs.spin_sum(wf1, wf2)
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
        wf2 = wfs.compute_wavefunction(s02, k1prime, k2prime, k3prime, x1, x2, x3)
        # Perform spin sum
        total += wfs.spin_sum(wf1, wf2)
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
        total += wfs.spin_sum(wf1, wf2)
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
- As for f_form_factor with s01 = - s02, the result is in general complex. For k_y = 0 it is real.
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
    gluon_sivers(k::Real;mu::Real=0.01,solver::String="cuhre")

Compute the gluon Sivers function for momentum transfer k

### Input

- `k`       -- Modulus of transverse momentum transfer
- `Δ`       -- Total momentum transfer
               - 2d real vector
- `mu`      -- Regulator for integrand
- `solver`  -- (optional, default =`"cuhre`) Integration strategy
               - Either "cuhre", "vegas", "divonne", "suave"

### Output

- Value of the gluon Sivers function at k

### Notes
- Assumes that 2d k vector is [kx, 0]. For k not along x,
  one would need to compute both s01, s02 = 1, -1 and -1, 1 and sum
  similarly to f2_form_factor.
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

- `mu`      -- Regulator for integration
- `solver`  -- (optional, default =`"cuhre`) Integration strategy
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
    write_odderon_distribution_range_to_csv(kmin::Real, kstep::Real, kmax::Real, mu::Real, [solver]::String="cuhre")

Write result of odderon_distribution for |k| in [kmin,kmax] GeV in steps of kstep GeV.

### Input
- `kmin`    -- Minimum value of k in GeV
- `kstep`   -- Step interval for k in GeV
- `kmax`    -- Maximun value of k in GeV
- `mu`      -- Regulator for integration
- `solver`  -- (optional, default =`"cuhre`) Integration strategy
              - Either "cuhre", "vegas", "divonne", "suave"
"""
function write_odderon_distribution_range_to_csv(kmin::Real, kstep::Real, kmax::Real, mu::Real, solver::String="cuhre")
    filename = "output_$(solver)_$(mu)_$(kmin)_$(kmax).csv"
    open(filename, "w") do io 
        println(io, "k,val_re,val_im,err_re,err_im,prob_re,prob_im,neval,fail,nregions")  # header
        for k in kmin:kstep:kmax
            # val = odderon_distribution(1,-1,[k,0],[0,0],mu)
            # println(io, "$(k),$(real(val)),$(imag(val))")
            integral, err, prob, neval, fail, nregions  = odderon_distribution(1, -1, [k,0], [0,0]; mu=mu, solver=solver)
            println(io, "$(k),$(integral[1]),$(integral[2]),$(err[1]),$(err[2]),$(prob[1]),$(prob[2]),$(neval),$(fail),$(nregions)")
            flush(io) 
        end
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