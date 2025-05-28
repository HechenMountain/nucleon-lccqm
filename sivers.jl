module Sivers

using Base: pi
using Printf
using PyPlot
using MCIntegration
using Cuba
# using SpecialFunctions
using Cubature

# Parallelization
using Base.Threads

# Progress bar
using ProgressMeter
const Atomic = Base.Threads.Atomic
const atomic_add! = Base.Threads.atomic_add!

export  gaussian, plot_gaussian,
        gaussian_3d, plot_gaussian_3d, SPIN_MAP,kronecker_delta,
        momentum_space_wavefunction, spin_wavefunction,
        baryon_wave_function, normalize_wave_function,
        f1_form_factor, f1_form_factor_table

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

kronecker_delta(a, b) = a == b ? 1 : 0

function spin_wavefunction(s0,s1,s2,s3,k1, k2, k3, x1, x2, x3;mq=0.26, β=0.55)
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


function momentum_space_wavefunction(k1, k2, k3, x1, x2, x3; mq = 0.26, β = 0.55)
    mq2 = mq^2
    m02 = (sum(k1.^2) + mq2)/x1 + (sum(k2.^2) + mq2)/x2 + (sum(k3.^2) + mq2)/x3
    return exp(-m02 / (2 * β^2))
end

function baryon_wave_function(s0,s1,s2,s3,k1, k2, k3, x1, x2, x3;norm=574.8236114423403,mq=0.26, β=0.55)
    ms_wf =  momentum_space_wavefunction(k1, k2, k3, x1, x2, x3; mq=mq, β=β)
    spin_wf = spin_wavefunction(s0,s1,s2,s3,k1,k2,k3,x1,x2,x3;mq=mq,β=β)
    wf = norm * ms_wf * spin_wf / sqrt(3)
    return wf
end

function normalize_wave_function(s0;mq=0.26, β=0.55)
    total = 0.0
    for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
        integrand(x,f) = begin
            # Cuba samples are [0,1]^n
            x1 = x[1]
            x2 = (1 - x1) * x[2]
            x3 = 1 - x1 - x2
            # Rescale to correct intervals [0,∞)
            k1 = [x[i]/(1 - x[i]) for i in 3:4]
            k2 = [x[i]/(1 - x[i]) for i in 5:6]
    
            k3 = k1 - k2  # Enforce transverse momentum conservation
            # Parton x is not transformed, but k_perp is
            dk = [1/(1 - x[i])^2 for i in 3:6]
        
            #wf = baryon_wave_function(s0,s1,s2,s3,k1,k2,k3,x1,x2,x3;mq=mq,β=β)
            ms_wf = momentum_space_wavefunction(k1, k2, k3, x1, x2, x3; mq=mq, β=β)
            spin_wf = spin_wavefunction(s0,s1,s2,s3,k1,k2,k3,x1,x2,x3;mq=mq,β=β) 
            wf = ms_wf * spin_wf
            f[1] = abs2(wf) * prod(dk)
        end
        result, err = cuhre(integrand, 6, 1, atol=1e-8, rtol=1e-6);
        # result, err = vegas(integrand, 6, 1, maxevals=5_000_000, nvec=10_000);
        total += result[1]
    end
    # println(" Result of cuhre: ", result[1], " ± ", err[1])
    # Integral is (-∞,+∞) for each k
    return 1/sqrt(1/(4π)/(2π)^2 * total*2^4)
end

function f1_form_factor(Δ;norm=574.8236114423403,mq=0.26, β=0.55)
    eu, ed = 2/3, -1/3
    charges = (eu,eu,ed)
    total = 0
    for (i,q) in enumerate(charges)
        for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
            integrand(x,f) = begin
                # Cuba samples are [0,1]^n
                x1 = x[1]
                x2 = (1 - x1) * x[2]
                x3 = 1 - x1 - x2
                # Rescale to correct intervals [0,∞)
                k1 = [x[i]/(1 - x[i]) for i in 3:4]
                k2 = [x[i]/(1 - x[i]) for i in 5:6]
        
                k3 = k1 - k2  # Enforce transverse momentum conservation

                k1prime = k1 - x1 * Δ + kronecker_delta(i,1) * Δ
                k2prime = k2 - x2 * Δ + kronecker_delta(i,2) * Δ
                k3prime = k3 - x3 * Δ + kronecker_delta(i,3) * Δ

                # Parton x is not transformed, but k_perp is
                dk = [1/(1 - x[i])^2 for i in 3:6]
                # Both wave functions have spin up
                wf1 = baryon_wave_function(1,s1,s2,s3,k1,k2,k3,x1,x2,x3;norm=norm,mq=mq,β=β)
                wf2 = baryon_wave_function(1,s1,s2,s3,k1prime,k2prime,k3prime,x1,x2,x3;norm=norm,mq=mq,β=β)

                f[1] = real(q * conj(wf2) * wf1 * prod(dk))
            end
            result, err = cuhre(integrand, 6, 1, atol=1e-12, rtol=1e-10);
            # result, err = vegas(integrand, 6, 1, maxevals=1_000_000, nvec=1_000);
            total += result[1]
        end
    end
    # Integral is (-∞,+∞) for each k
    return 3 / (4π) / (2π)^2 * total * 2^4
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

"""
    gaussian(x, norm, width)

Returns the value of a Gaussian function at position `x`, with normalization `norm` and width `width`.
The function is defined as:

    norm * exp(-x^2 / (2 * width^2))
"""

function gaussian(x,norm,width)
    return norm * exp(-x^2/(2*width^2))
end
"""
    gaussian_3d(x, norm, width)

Returns the value of a 3D Gaussian function at position `x, y, z`, with normalization `norm` and width `width_x, width_y, width_z`.
The function is defined as:

    norm * exp(-x^2/(2*width_x) - y^2/(2*width_y) - z^2/(2*width_z))
"""

function gaussian_3d(x,y,z,norm,width_x,width_y,width_z)
    exponent =  x^2/(2*width_x) + y^2/(2*width_y) + z^2/(2*width_z)
    return norm * exp(-exponent)
end

"""
    plot_gaussian(norm, width; xrange=(-5, 5), n=1000)

Plots the Gaussian with given normalization and width over the specified range.
"""

function plot_gaussian(norm,width;xrange=(-5,5), n=1000)
    x_values = range(xrange[1],xrange[2],length=n)
    y_values = [gaussian(x,norm,width) for x in x_values]
    figure()
    plot(x_values,y_values,label=L"A\cdot e^{(-x^2)}")
    xlabel(L"x")
    ylabel(L"f(x)")
    legend()
    grid(true)
end

"""
    plot_gaussian_3d(norm, width_x, width_y, width_z; range=(-3, 3), n=50)

Plots a 3D Gaussian over x, y, z using surface slices in each plane.
"""

function plot_gaussian_3d(norm, width_x, width_y, width_z; rspan=(-3, 3), n=50)
    x = range(rspan[1], rspan[2], length=n)
    y = range(rspan[1], rspan[2], length=n)
    z = range(rspan[1], rspan[2], length=n)

    # Meshgrid for x-y slice at z=0
    X = [x[i] for i in 1:n, j in 1:n]
    Y = [y[j] for i in 1:n, j in 1:n]
    Zxy = [gaussian_3d(X[i,j], Y[i,j], 0.0, norm, width_x, width_y, width_z) for i in 1:n, j in 1:n]

    fig = figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Zxy, cmap="viridis", alpha=0.8)

    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.set_zlabel(L"f(x, y, z{=}0)")
    ax.set_title(L"\text{3D Gaussian slice at } z=0")
    fig.tight_layout()
end

end # module