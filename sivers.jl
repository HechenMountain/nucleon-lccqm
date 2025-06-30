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

# export  SPIN_MAP,kronecker_delta,
#         momentum_space_wavefunction, spin_wavefunction,
#         baryon_wavefunction, normalize_wavefunction,
#         f1_form_factor, f1_form_factor_table,cubic_color_corellator,gluon_sivers_direct

export  SPIN_MAP,kronecker_delta,
        momentum_space_wavefunction, spin_wavefunction,
        baryon_wavefunction#, normalize_wavefunction

# All possible proton and constituent quar spin configurations
# Eq.(22) and (23) in the draft
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

function spin_wavefunction(s0,s1,s2,s3,k1, k2, k3, x1, x2, x3)
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


function momentum_space_wavefunction(k1, k2, k3, x1, x2, x3)
    # Parameters
    mq = params.mq
    β = params.β

    mq2 = mq^2
    m02 = (sum(k1.^2) + mq2)/x1 + (sum(k2.^2) + mq2)/x2 + (sum(k3.^2) + mq2)/x3
    return exp(-m02 / (2 * β^2))
end

function baryon_wavefunction(s0,s1,s2,s3,k1, k2, k3, x1, x2, x3)
    # Parameters
    norm = params.norm

    ms_wf =  momentum_space_wavefunction(k1, k2, k3, x1, x2, x3)
    spin_wf = spin_wavefunction(s0,s1,s2,s3,k1,k2,k3,x1,x2,x3)
    wf = norm * ms_wf * spin_wf / sqrt(3)
    return wf
end

# Currently the summation is done inside the integrand
# such that cuhre is only called once.
# However, the speed-up is only mild since
# the integrand is better behaved when individual
# contributions are integrated first.
# We might want to keep this in mind if 
# higher precision is desired.
function normalize_wavefunction(s0)
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
        dk1 = 1/(1 - x[3])^2 # r1
        dk2 = 1/(1 - x[5])^2 # r2
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

function f1_form_factor(Δ)
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
        dk1 = 1/(1 - x[3])^2 # r1
        dk2 = 1/(1 - x[5])^2 # r2
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

        f[1] = real(total * dk * dϕ)
    end
    integral, err = cuhre(integrand, 6, 1, atol=1e-12, rtol=1e-10);
    result =  3 / (4π) / (2π)^2 * integral[1]
    return result
end

function f1_form_factor_table(Δ_array)
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

function cubic_color_corellator(s01,s02,q1,q2,q3)

    # Essentially as in f1_form_factor but with different kinematics
    # placeholder for dabc
    dabc = 1
    # Δ determined from overall momentum conservation
    Δ = - (q1 + q2 + q3)
    total = 0
    # It seems cuhre is faster with the sum outside the integrand
    for j1 in 1:3, j2 in 1:3, j3 in 1:3
        if j1 == j2 || j1 == j3 || j2 == j3
            continue
        end
        for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
            function integrand(x,f)
                # Cuba samples are [0,1]^n
                x1 = x[1]
                x2 = (1 - x1) * x[2]
                x3 = 1 - x1 - x2
                # Rescale to correct intervals [0,∞)
                k1 = [x[j]/(1 - x[j]) for j in 3:4]
                k2 = [x[j]/(1 - x[j]) for j in 5:6]
        
                k3 = k1 - k2  # Enforce transverse momentum conservation
                # Momentum inflow from each attached vertex
                delta_terms =  kronecker_delta(1, j1) * q1 - kronecker_delta(2, j2) * q2 - kronecker_delta(3, j3) * q3
                # Residual kinematics fixed by parton x
                k1prime = k1 - x1 * Δ - delta_terms
                k2prime = k2 - x2 * Δ - delta_terms
                k3prime = k3 - x3 * Δ - delta_terms

                # Parton x is not transformed, but k_perp is
                dk = [1/(1 - x[j])^2 for j in 3:6]
                # Both wave functions have spin up
                wf1 = baryon_wavefunction(s01,s1,s2,s3,k1,k2,k3,x1,x2,x3)
                wf2 = baryon_wavefunction(s02,s1,s2,s3,k1prime,k2prime,k3prime,x1,x2,x)

                f[1] = real(conj(wf2) * wf1 * prod(dk))
            end
            result, err = cuhre(integrand, 6, 1, atol=1e-12, rtol=1e-10);
            total += result[1]
        end
    end
    # Integral is (-∞,+∞) for each k so we get 2^4
    total = - 0.25 * dabc * 3 / (4π) / (2π)^2 * total * 2^4
    return total
end

function three_gluon_amplitude(s01,s02,r_perp,b_perp,Δ)
    # Eq. 1 in chic_sivers-notes or D11 in 2402.19134, respectively
    # prf is a placeholder for stuff like g³, Nc...
    prf = 1
    function integrand(x,f)
        # Here I'm not so sure about the symmetry
        # But we could probably utilize it
        q1 = [tan(pi * (x[j] - 0.5)) for j in 1:2]
        q2 = [tan(pi * (x[j] - 0.5)) for j in 3:4]
        q3 = [tan(pi * (x[j] - 0.5)) for j in 5:6]
        # Jacobian
        dq = [pi * sec(pi * (x[j] - 0.5))^2 for j in 1:6]

        ccc = cubic_color_corellator(s01,s02,q1,q2,q3) 
        den = sum(q1.^2) * sum(q1.^2) * sum(q3.^2)
        trig1 = sin(dot(b_perp, q1 + q2 + q3))
        trig2 = sin(dot(r_perp/2, q1 - q2 - q3))
        trig3 = sin(dot(r_perp/2, q1 + q2 + q3))
        f[1] = ccc / den * trig1 * (trig2 + 1/3 * trig3) * prod(dq)
    end
    result, err = cuhre(integrand, 6, 1, atol=1e-12, rtol=1e-10);
    return prf * result
end

function ft_three_gluon_amplitude(s01,s02,k_perp,Δ)
    # Eq.13 in chic_sivers-notes
    # FT over r_perp and b_perp to get 
    # k_perp and Delta dependence 
    function integrand(x,f)
        # Again, not sure about the correct symmetry
        # r_perp, b_perp ∈ (-∞, ∞)
        r_perp = [tan(pi * (x[j] - 0.5)) for j in 1:2]
        dr_perp = [pi * sec(pi * (x[j] - 0.5))^2 for j in 1:2]

        b_perp = [tan(pi * (x[j] - 0.5)) for i in 3:4]
        db_perp = [pi * sec(pi * (x[j] - 0.5))^2 for j in 3:4]

        exponent = exp( - 1im * dot(k_perp,r_perp) - 1im * dot(Δ,b_perp))
        ggg = three_gluon_amplitude(s01,s02,r_perp,b_perp,Δ)

        ft_ggg_integrand = exponent * ggg * prod(dr_perp) * prod(db_perp)
        f[1] = real(ft_ggg_integrand)
    end
    result, err = cuhre(integrand, 4, 1, atol=1e-10, rtol=1e-8)
    return result[1]
end

function gluon_sivers_direct(k_perp)
    dabc = 1
    prf = 1
    total = 0
    # The forward part with spin flip gives the gluon sivers function
    Δ = [0,0]
    s01 = 1
    s02 = -1
    # It seems cuhre is faster with the sum outside the integrand
    for j1 in 1:3, j2 in 1:3, j3 in 1:3
        if j1 == j2 || j1 == j3 || j2 == j3
            continue
        end
        for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
            function integrand(x, f)
                # Integration variables
                x1 = x[1]
                x2 = (1 - x1) * x[2]
                x3 = 1 - x1 - x2

                k1 = [tan(pi * (x[j] - 0.5)) for j in 3:4]
                k2 = [tan(pi * (x[j] - 0.5)) for j in 5:6]
                k3 = k1 - k2
                dk = [1 / (1 - x[j])^2 for j in 3:6]

                q1 = [tan(pi * (x[j] - 0.5)) for j in 7:8]
                q2 = [tan(pi * (x[j] - 0.5)) for j in 9:10]
                q3 = [tan(pi * (x[j] - 0.5)) for j in 11:12]
                dq = [pi * sec(pi * (x[j] - 0.5))^2 for j in 7:12]

                r_perp = [tan(pi * (x[j] - 0.5)) for j in 13:14]
                dr_perp = [pi * sec(pi * (x[j] - 0.5))^2 for j in 13:14]

                b_perp = [tan(pi * (x[j] - 0.5)) for j in 15:16]
                db_perp = [pi * sec(pi * (x[j] - 0.5))^2 for j in 15:16]

                # cubic_color_corellator part
                delta_terms = kronecker_delta(1, j1) * q1 - kronecker_delta(2, j2) * q2 - kronecker_delta(3, j3) * q3
                k1prime = k1 - x1 * Δ - delta_terms
                k2prime = k2 - x2 * Δ - delta_terms
                k3prime = k3 - x3 * Δ - delta_terms
                
                wf1 = baryon_wavefunction(s01, s1, s2, s3, k1, k2, k3, x1, x2, x3)
                wf2 = baryon_wavefunction(s02, s1, s2, s3, k1prime, k2prime, k3prime, x1, x2, x3)
                ccc = real(conj(wf2) * wf1 * prod(dk))

                # three gluon three_gluon_amplitude
                den = sum(q1.^2) * sum(q2.^2) * sum(q3.^2)
                trig1 = sin(dot(b_perp, q1 + q2 + q3))
                trig2 = sin(dot(r_perp / 2, q1 - q2 - q3))
                trig3 = sin(dot(r_perp / 2, q1 + q2 + q3))
                three_gluon = ccc / den * trig1 * (trig2 + 1/3 * trig3) * prod(dq)

                # Fourier Transform
                exponent = exp(-1im * dot(k_perp, r_perp) - 1im * dot(Δ, b_perp))

                f[1] = real(three_gluon * exponent * prod(dr_perp) * prod(db_perp))
            end

            result, err = cuhre(integrand, 16, 1, atol=1e-6, rtol=1e-6)
            total += result[1]
        end
    end

    total = dabc * prf * total
    return total
end

function gluon_sivers(k_perp)
    # Eq.19 in chic_sivers-notes
    # We fix Delta = q1 + q2 + q3 = [0,0] since we are interested in the forward limit
    # and take the spin-flip part to get the gluon Sivers function
    s01 = 1
    s02 = -1
    result = ft_three_gluon_amplitude(s01,s02,k_perp,0)
    return result 
end

end # module