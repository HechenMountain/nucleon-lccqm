# This code performs parallel computation of the electromagnetic form factors,
# odderon distribution and gluon Sivers function
# over a specified range of Δ and k values.
# The results are stored in CSV files in the ../data folder.
# Just uncomment the function calls at the bottom to use.
# Run for example over command line from project root via:
# nohup JULIA_PROJECT=@. julia -p nworkers scripts/writers.jl > writers.log 2>&1 &
# Where nworkers is the number of parallel workers to use.

# ======================
# Environment Setup
# ======================

using Pkg
# Activate the Sivers project environment once on the main process
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Distributed
using Dates
using SharedArrays

# Load Sivers package on all workers
println(Dates.now(), " Starting module import for $(nworkers()) workers...")
flush(stdout)

@everywhere begin
    using GluonSiversLCCQM
    using SharedArrays
end

println(Dates.now(), " Finished module import.")

# ======================
# Data Directory Setup
# ======================

"""
    data_dir(subdir::String="csv")

Get the path to the data output directory.
Creates the directory if it doesn't exist.

Arguments
- `subdir`: subdirectory name ("csv" or "plots")

Returns
- Absolute path to the data subdirectory
"""
function data_dir(subdir::String="csv")
    dir = joinpath(dirname(@__DIR__), "data", subdir)
    isdir(dir) || mkpath(dir)
    return dir
end
flush(stdout)

# ======================
# Writers
# ======================

"""
    write_cuba_to_file(filename::String, results::SharedMatrix{Float64})

Write results from a computation to a CSV file.

Arguments
- `filename`: output CSV filename
- `results`: shared matrix with rows `[k, val_re, val_im, err_re, err_im, prob_re, prob_im, neval, fail, nregions]`

Returns
- Nothing. Writes the CSV file.
"""
function write_cuba_to_file(filename::String,results::SharedMatrix{Float64})
    n = length(results[:,1])
    open(filename, "w") do io
    println(io, "k,val_re,val_im,err_re,err_im,prob_re,prob_im,neval,fail,nregions")
    for i in 1:n
        k = results[i,1]
        integral_re = results[i,2]
        integral_im = results[i,3]
        err_re = results[i,4]
        err_im = results[i,5]
        prob_re = results[i,6]
        prob_im = results[i,7]
        neval = Int(results[i,8])
        fail = Int(results[i,9])
        nregions = Int(results[i,10])
        println(io, "$(k),$(integral_re),$(integral_im),$(err_re),$(err_im),$(prob_re),$(prob_im),$(neval),$(fail),$(nregions)")
        flush(io)
        end
    end
    flush(stdout) # Ensure print statements are output
end

"""
        write_2d_cuba_to_file(filename::String, k_list::AbstractVector{<:AbstractVector}, results::AbstractMatrix{Float64})

Write a CSV for 2D k-grid results with columns `k_x,k_y,val_re,val_im,err_re,err_im,prob_re,prob_im,neval,fail,nregions`.

Arguments
- `filename`: output CSV filename
- `k_list`: array of `[kx, ky]` pairs, one per row of `results`
- `results`: numeric matrix mirroring `write_cuba_to_file` column order; column 1 may hold |k|

Returns
- Nothing. Writes the CSV file.
"""
function write_2d_cuba_to_file(filename::String, k_list::AbstractVector{<:AbstractVector}, results::AbstractMatrix{Float64})
    n = length(k_list)
    open(filename, "w") do io
        # Header matches write_cuba_to_file but with k_x,k_y instead of single k
        println(io, "k_x,k_y,val_re,val_im,err_re,err_im,prob_re,prob_im,neval,fail,nregions")
        for i in 1:n
            kx = k_list[i][1]
            ky = k_list[i][2]
            # results columns mirror write_cuba_to_file where col 1 was k (magnitude)
            # and cols 2..10 correspond to val_re,val_im,err_re,err_im,prob_re,prob_im,neval,fail,nregions
            val_re = results[i,2]
            val_im = results[i,3]
            err_re = results[i,4]
            err_im = results[i,5]
            prob_re = results[i,6]
            prob_im = results[i,7]
            neval = Int(results[i,8])
            fail = Int(results[i,9])
            nregions = Int(results[i,10])
            println(io, "$(kx),$(ky),$(val_re),$(val_im),$(err_re),$(err_im),$(prob_re),$(prob_im),$(neval),$(fail),$(nregions)")
            flush(io)
        end
    end
end

"""
    write_gluon_sivers_to_csv(kmin::Real, kmax::Real, n::Integer; μ::Real, solver::Symbol=:vegas, spacing::Symbol=:lin)

Write gluon Sivers values for |k| ∈ [kmin, kmax] into a CSV.

Arguments
- `kmin`: minimum k in GeV
- `kmax`: maximum k in GeV
- `n`: number of k points
- `μ`: regulator for the integration
- `solver`: integration backend (:vegas default, or :cuhre, :divonne, :suave)
- `spacing`: spacing type (:lin default or :log)

Returns
- Nothing. Writes `output_gluon_sivers_*.csv`.
"""
function write_gluon_sivers_to_csv(kmin::Real, kmax::Real, n::Integer;
                                   μ::Real, solver::Symbol=:vegas, spacing::Symbol=:lin)
    if spacing == :log
        k_list = 10 .^ collect(range(log10(kmin), log10(kmax), length=n))
    elseif spacing == :lin
        k_list = collect(range(kmin, stop=kmax, length=n))
    else
        error("Invalid spacing option: $spacing. Use :lin or :log.")
    end
    # columns: k, val_re, val_im, err_re, err_im, prob_re, prob_im, neval, fail, nregions
    results = SharedArray{Float64}(n,10)

    @sync @distributed for i in 1:n
        k = k_list[i]
        integral, err, prob, neval, fail, nregions = Sivers.gluon_sivers(k; μ=μ, solver=solver)
        results[i, :] .= (Float64(k), Float64(integral[1]), Float64(integral[2]), 
                          Float64(err[1]), Float64(err[2]), Float64(prob[1]), Float64(prob[2]),
                          Float64(neval), Float64(fail), Float64(nregions))
    end
    # Write to file
    filename = joinpath(data_dir("csv"), "output_gluon_sivers_$(solver)_$(μ)_$(kmin)_$(kmax).csv")
    println(Dates.now(), " Writing to file ", filename)
    write_cuba_to_file(filename,results)
end

"""
    write_2d_odderon_distribution_to_csv(s01::Integer, s02::Integer, kmin::Real, kmax::Real, n::Integer; μ::Real, solver::Symbol=:vegas, spacing::Symbol=:lin)

Write odderon_distribution on a 2D k grid into a CSV.

Arguments
- `s01, s02`: proton spins (+1 or -1)
- `kmin`: minimum |k| in GeV
- `kmax`: maximum |k| in GeV
- `n`: number of points per side (positive and negative axes)
- `μ`: regulator for the integration
- `solver`: integration backend (:vegas default, or :cuhre, :divonne, :suave)
- `spacing`: spacing type (:lin default or :log)

Returns
- Nothing. Writes `output_2d_*.csv`.
"""
function write_2d_odderon_distribution_to_csv(s01::Integer,s02::Integer,kmin::Real, kmax::Real, n::Integer;
                                              μ::Real, solver::Symbol=:vegas, spacing::Symbol=:lin)
    # Build 1D k values excluding the central region [-kmin, kmin].
    # We take negative side from -kmax to -kmin and positive side from kmin to kmax.
    if spacing == :log
        neg_vals = -10 .^ collect(range(log10(kmax), log10(kmin), length=n))
        pos_vals = 10 .^ collect(range(log10(kmin), log10(kmax), length=n))
    elseif spacing == :lin
        neg_vals = collect(range(-kmax, stop=-kmin, length=n))
        pos_vals = collect(range(kmin, stop=kmax, length=n))
    else
        error("Invalid spacing option: $spacing. Use :lin or :log.")
    end
    k_vals = vcat(neg_vals, pos_vals)
    # Create 2D list of [kx, ky] pairs covering the Cartesian product of k_vals.
    k_list = [[kx, ky] for kx in k_vals for ky in k_vals]
    # columns: k, val_re, val_im, err_re, err_im, prob_re, prob_im, neval, fail, nregions
    results = SharedArray{Float64}(n,10)

    @sync @distributed for i in 1:n
        k = k_list[i]
        # integral, err, prob, neval, fail, nregions = test(k)
        integral, err, prob, neval, fail, nregions = Sivers.odderon_distribution(s01, s02, [0,0], k; μ=μ, solver=solver)
        # store magnitude |k| in the first column (existing CSV expects a scalar k)
        k_mag = hypot(k[1], k[2])
        results[i, :] .= (Float64(k_mag), Float64(integral[1]), Float64(integral[2]), 
                          Float64(err[1]), Float64(err[2]), Float64(prob[1]), Float64(prob[2]),
                          Float64(neval), Float64(fail), Float64(nregions))
    end
    # Write to file
    filename = joinpath(data_dir("csv"), "output_2d_$(s01)_$(s02)_$(solver)_$(μ)_$(kmin)_$(kmax).csv")
    println(Dates.now(), " Writing to file ", filename)
    write_2d_cuba_to_file(filename,k_list,results)
end


"""
    write_odderon_distribution_to_csv(kmin::Real, kmax::Real, n::Integer; μ::Real, solver::Symbol=:vegas, spacing::Symbol=:lin)

Write odderon_distribution for |k| ∈ [kmin, kmax] into a CSV.

Arguments
- `kmin`: minimum k in GeV
- `kmax`: maximum k in GeV
- `n`: number of k points
- `μ`: regulator for the integration
- `solver`: integration backend (:vegas default, or :cuhre, :divonne, :suave)
- `spacing`: spacing type (:lin default or :log)

Returns
- Nothing. Writes `output_*.csv`.
"""
function write_odderon_distribution_to_csv(kmin::Real, kmax::Real, n::Integer;
                                           μ::Real, solver::Symbol=:vegas, spacing::Symbol=:lin)
    if spacing == :log
        k_list = 10 .^ collect(range(log10(kmin), log10(kmax), length=n))
    elseif spacing == :lin
        k_list = collect(range(kmin, stop=kmax, length=n))
    else
        error("Invalid spacing option: $spacing. Use :lin or :log.")
    end
    # columns: k, val_re, val_im, err_re, err_im, prob_re, prob_im, neval, fail, nregions
    results = SharedArray{Float64}(n,10)

    @sync @distributed for i in 1:n
        k = k_list[i]
        integral, err, prob, neval, fail, nregions = Sivers.odderon_distribution(1, -1, [0,0], [k,0]; μ=μ, solver=solver)
        results[i, :] .= (Float64(k), Float64(integral[1]), Float64(integral[2]), 
                          Float64(err[1]), Float64(err[2]), Float64(prob[1]), Float64(prob[2]),
                          Float64(neval), Float64(fail), Float64(nregions))
    end
    # Write to file
    filename = joinpath(data_dir("csv"), "output_$(solver)_$(μ)_$(kmin)_$(kmax).csv")
    println(Dates.now(), " Writing to file ", filename)
    write_cuba_to_file(filename,results)
end

"""
    write_odderon_distribution_r_to_csv(rmin::Real, rmax::Real, n::Integer; μ::Real, solver::Symbol=:vegas, spacing::Symbol=:lin)

Write odderon_distribution_r for |r| ∈ [rmin, rmax] (GeV⁻¹) into a CSV.

Arguments
- `rmin`: minimum r in GeV⁻¹
- `rmax`: maximum r in GeV⁻¹
- `n`: number of r points
- `μ`: regulator for the integration
- `solver`: integration backend (:vegas default, or :cuhre, :divonne, :suave)
- `spacing`: spacing type (:lin default or :log)

Returns
- Nothing. Writes `output_r_*.csv`.
"""
function write_odderon_distribution_r_to_csv(rmin::Real, rmax::Real, n::Integer;
                                            μ::Real, solver::Symbol=:vegas, spacing::Symbol=:lin)
    if spacing == :log
        r_list = 10 .^ collect(range(log10(rmin), log10(rmax), length=n))
    elseif spacing == :lin
        r_list = collect(range(rmin, stop=rmax, length=n))
    else
        error("Invalid spacing option: $spacing. Use :lin or :log.")
    end
    # columns: r, val_re, val_im, err_re, err_im, prob_re, prob_im, neval, fail, nregions
    results = SharedArray{Float64}(n,10)

    @sync @distributed for i in 1:n
        r = r_list[i]
        integral, err, prob, neval, fail, nregions = Sivers.odderon_distribution_r(1, -1, [0,0], [r,0]; μ=μ, solver=solver)
        results[i, :] .= (Float64(r), Float64(integral[1]), Float64(integral[2]), 
                          Float64(err[1]), Float64(err[2]), Float64(prob[1]), Float64(prob[2]),
                          Float64(neval), Float64(fail), Float64(nregions))
    end
    # Write to file
    filename = joinpath(data_dir("csv"), "output_r_$(solver)_$(μ)_$(rmin)_$(rmax).csv")
    println(Dates.now(), " Writing to file ", filename)
    write_cuba_to_file(filename,results)
end

"""
    write_cubic_color_corellator_to_csv(s01::Integer, s02::Integer,
                                         Δmin::Real, Δmax::Real, n::Integer,
                                         q12::Real, q23::Real;
                                         solver::Symbol=:vegas, spacing::Symbol=:lin)

Write integrated cubic color correlator values for |Δ| ∈ [Δmin, Δmax] into a CSV.

Arguments
- `s01, s02`: proton spins (+1 or -1)
- `Δmin`, `Δmax`: Δ range in GeV
- `n`: number of Δ points
- `q12, q23`: q1 - q2 and q2 - q3 values (assumed along x)
- `solver`: integration backend (:vegas default, or :cuhre, :divonne, :suave)
- `spacing`: spacing type (:lin default or :log)

Returns
- Nothing. Writes `output_ccc_*.csv`.

Notes
- Assumes momentum transfer along x.
"""
function write_cubic_color_corellator_to_csv(s01::Integer,s02::Integer,
                                             Δmin::Real, Δmax::Real, n::Integer,
                                             q12::Real, q23::Real;
                                             solver::Symbol=:vegas, spacing::Symbol=:lin)
    # Convert momenta to vectors
    q12, q23 = [q12,0], [q23,0]
    # Build momentum transfer range
    if spacing == :log
        Δ_list = 10 .^ collect(range(log10(Δmin), log10(Δmax), length=n))
    elseif spacing == :lin
        Δ_list = collect(range(Δmin, stop=Δmax, length=n))
    else
        error("Invalid spacing option: $spacing. Use :lin or :log.")
    end
    # Initialize results array
    # columns: Δ, val_re, val_im, err_re, err_im, prob_re, prob_im, neval, fail, nregions
    results = SharedArray{Float64}(n,10)

    @sync @distributed for i in 1:n
        Δ = [Δ_list[i],0]
        q1, q2, q3 = (2 * q12 + q23 - Δ) / 3, (- q12 + q23 - Δ) / 3, - (q12 + 2 * q23 + Δ) / 3
        integral, err, prob, neval, fail, nregions = Sivers.integrate_cubic_color_correlator(s01, s02, q1, q2, q3; solver=solver)
        results[i, :] .= (Float64(Δ_list[i]), Float64(integral[1]), Float64(integral[2]), 
                          Float64(err[1]), Float64(err[2]), Float64(prob[1]), Float64(prob[2]),
                          Float64(neval), Float64(fail), Float64(nregions))
    end
    # Write to file
    filename = joinpath(data_dir("csv"), "output_ccc_s01_$(s01)_s02_$(s02)_$(solver)_q12_$(q12)_q23_$(q23)_$(Δmin)_$(Δmax).csv")
    println(Dates.now(), " Writing to file ", filename)
    write_cuba_to_file(filename,results)
end

"""
    write_ft_cubic_color_corellator_to_csv(s01::Integer, s02::Integer,
                                           rmin::Real, rmax::Real, n::Integer;
                                           solver::Symbol=:vegas, spacing::Symbol=:lin)

Write Fourier-transformed cubic color correlator for |r| ∈ [rmin, rmax] into a CSV.

Arguments
- `s01, s02`: proton spins (+1 or -1)
- `rmin`, `rmax`: r range in GeV⁻¹
- `n`: number of r points
- `solver`: integration backend (:vegas default, or :cuhre, :divonne, :suave)
- `spacing`: spacing type (:lin default or :log)

Returns
- Nothing. Writes `output_ft_ccc_*.csv`.
"""
function write_ft_cubic_color_corellator_to_csv(s01::Integer,s02::Integer,
                                                rmin::Real, rmax::Real, n::Integer;
                                                solver::Symbol=:vegas, spacing::Symbol=:lin)
    if spacing == :log
        r_list = 10 .^ collect(range(log10(rmin), log10(rmax), length=n))
    elseif spacing == :lin
        r_list = collect(range(rmin, stop=rmax, length=n))
    else
        error("Invalid spacing option: $spacing. Use :lin or :log.")
    end
    n = length(r_list)
    # columns: k, val_re, val_im, err_re, err_im, prob_re, prob_im, neval, fail, nregions
    results = SharedArray{Float64}(n,10)

    @sync @distributed for i in 1:n
        r = r_list[i]
        integral, err, prob, neval, fail, nregions = Sivers.ft_cubic_color_correlator(s01, s02, r; solver=solver)
        results[i, :] .= (Float64(r_list[i]), Float64(integral[1]), Float64(integral[2]), 
                          Float64(err[1]), Float64(err[2]), Float64(prob[1]), Float64(prob[2]),
                          Float64(neval), Float64(fail), Float64(nregions))
    end
    # Write to file
    filename = joinpath(data_dir("csv"), "output_ft_ccc_s01_$(s01)_s02_$(s02)_$(solver)_$(rmin)_$(rmax).csv")
    println(Dates.now(), " Writing to file ", filename)
    write_cuba_to_file(filename,results)
end

"""
    write_f1_form_factor_to_csv(Δmin::Real=0.0, Δmax::Real=3.3, n::Integer=27; spacing::Symbol=:lin)

Write F1 form factor values for |Δ| ∈ [Δmin, Δmax] into a CSV.

Arguments
- `Δmin`, `Δmax`: Δ range in GeV
- `n`: number of Δ points
- `spacing`: spacing type (:lin default or :log)

Returns
- Nothing. Writes `output_f1.csv`.
"""
function write_f1_form_factor_to_csv(Δmin::Real=0.0, Δmax::Real=3.3, n::Integer=27; spacing::Symbol=:lin)
    if spacing == :log
        Δ_list = 10 .^ collect(range(log10(Δmin), log10(Δmax), length=n))
    elseif spacing == :lin
        Δ_list = collect(range(Δmin, stop=Δmax, length=n))
    else
        error("Invalid spacing option: $spacing. Use :lin or :log.")
    end
    n = length(Δ_list)
    # columns: Δ, val_re, val_im, err_re, err_im, prob_re, prob_im, neval, fail, nregions
    results = SharedArray{Float64}(n,10)

    @sync @distributed for i in 1:n
        Δ = Δ_list[i]
        integral, err, prob, neval, fail, nregions = Sivers.f1_form_factor([Δ, 0])
        results[i, :] .= (Float64(Δ), Float64(integral[1]), Float64(integral[2]),
                          Float64(err[1]), Float64(err[2]), Float64(prob[1]), Float64(prob[2]),
                          Float64(neval), Float64(fail), Float64(nregions))
    end
    # Write to file
    filename = joinpath(data_dir("csv"), "output_f1.csv")
    println(Dates.now(), " Writing to file ", filename)
    write_cuba_to_file(filename,results)
end

"""
    write_f2_form_factor_to_csv(Δmin::Real=1e-6, Δmax::Real=3.3, n::Integer=27; spacing::Symbol=:lin)

Write F2 form factor values for |Δ| ∈ [Δmin, Δmax] into a CSV.

Arguments
- `Δmin`, `Δmax`: Δ range in GeV
- `n`: number of Δ points
- `spacing`: spacing type (:lin default or :log)

Returns
- Nothing. Writes `output_f2.csv`.
"""
function write_f2_form_factor_to_csv(Δmin::Real=1e-6, Δmax::Real=3.3, n::Integer=27; spacing::Symbol=:lin)
    if spacing == :log
        Δ_list = 10 .^ collect(range(log10(Δmin), log10(Δmax), length=n))
    elseif spacing == :lin
        Δ_list = collect(range(Δmin, stop=Δmax, length=n))
    else
        error("Invalid spacing option: $spacing. Use :lin or :log.")
    end
    n = length(Δ_list)
    # columns: Δ, val_re, val_im, err_re, err_im, prob_re, prob_im, neval, fail, nregions
    results = SharedArray{Float64}(n,10)

    @sync @distributed for i in 1:n
        Δ = Δ_list[i]
        integral, err, prob, neval, fail, nregions = Sivers.f2_form_factor([Δ, 0])
        results[i, :] .= (Float64(Δ), Float64(integral[1]), Float64(integral[2]), 
                          Float64(err[1]), Float64(err[2]), Float64(prob[1]), Float64(prob[2]),
                          Float64(neval), Float64(fail), Float64(nregions))
    end
    # Write to file
    filename = joinpath(data_dir("csv"), "output_f2.csv")
    println(Dates.now(), " Writing to file ", filename)
    write_cuba_to_file(filename,results)
end

"""
    write_f_form_factor_to_csv(s01::Integer, s02::Integer, Δmin::Real=1e-6, Δmax::Real=3.3, n::Integer=27; spacing::Symbol=:lin)

Write general F-type form factor for |Δ| ∈ [Δmin, Δmax] into a CSV.

Arguments
- `s01, s02`: ingoing/outgoing proton spins (+1 or -1)
- `Δmin`, `Δmax`: Δ range in GeV
- `n`: number of Δ points
- `spacing`: spacing type (:lin default or :log)

Returns
- Nothing. Writes `output_f_*.csv`.
"""
function write_f_form_factor_to_csv(s01::Integer,s02::Integer,Δmin::Real=1e-6, Δmax::Real=3.3, n::Integer=27; spacing::Symbol=:lin)
    if spacing == :log
        Δ_list = 10 .^ collect(range(log10(Δmin), log10(Δmax), length=n))
    elseif spacing == :lin
        Δ_list = collect(range(Δmin, stop=Δmax, length=n))
    else
        error("Invalid spacing option: $spacing. Use :lin or :log.")
    end
    # columns: Δ, val_re, val_im, err_re, err_im, prob_re, prob_im, neval, fail, nregions
    results = SharedArray{Float64}(n,10)

    @sync @distributed for i in 1:n
        Δ = Δ_list[i]
        integral, err, prob, neval, fail, nregions = Sivers.f_form_factor(s01,s02,[Δ, 0])
        results[i, :] .= (Float64(Δ), Float64(integral[1]), Float64(integral[2]), 
                          Float64(err[1]), Float64(err[2]), Float64(prob[1]), Float64(prob[2]),
                          Float64(neval), Float64(fail), Float64(nregions))
    end
    # Write to file
    filename = joinpath(data_dir("csv"), "output_f_$(s01)_$(s02).csv")
    println(Dates.now(), " Writing to file ", filename)
    write_cuba_to_file(filename,results)
end

# ======================
# Main calls
# ======================

println(Dates.now(), " Starting writers...",)
flush(stdout)
# Add calls to the functions here as desired
# write_f1_form_factor_to_csv()
# write_f2_form_factor_to_csv()
# write_f_form_factor_to_csv(1, -1)
# write_2d_odderon_distribution_to_csv(1,-1, 1e-4, 1.001, 21; μ=0.0, solver=:vegas)  # 0.05 step ≈ 21 points per side (kmin,kmax,n)
# write_2d_odderon_distribution_to_csv(-1,1, 1e-4, 1.001, 21; μ=0.0, solver=:vegas)
# write_2d_odderon_distribution_to_csv(1,-1, 1e-4, 0.25, 25; μ=0.0, solver=:vegas)  # 0.01 step ≈ 25 points per side (kmin,kmax,n)

# Single function calls in parallel using pmap
# xs = [
#     (1,-1,[0,0],[.610,0]),
#     (1,-1,[0,0],[0.5001,0.3501])
# ]

# results = pmap(xs) do (a,b,c,d)
#     Sivers.odderon_distribution(a,b,c,d)
# end

# for r in results
#     println(r)
# end
# println(Sivers.odderon_distribution(1,-1,[0,0],[0.5001,0.3501]))
# write_cubic_color_corellator_to_csv(1, 1, 1e-4, 10.001, 100, 0, 0; solver=:vegas)  # (Δmin, Δmax, n, q12, q23)
# write_cubic_color_corellator_to_csv(1, 1, 10.01, 30.0, 200, 0, 0; solver=:vegas) 
# write_cubic_color_corellator_to_csv(1, -1, 1e-4, 10.001, 100, 0, 0; solver=:vegas)

write_odderon_distribution_r_to_csv(0.01, 1, 36; μ=0, solver=:vegas,spacing=:log) # <--- Currently running writers.log
write_odderon_distribution_r_to_csv(1.01, 4, 36; μ=0, solver=:vegas)
write_odderon_distribution_r_to_csv(4.01, 7, 36; μ=0, solver=:vegas)
write_odderon_distribution_r_to_csv(7.01, 10, 36; μ=0, solver=:vegas)
write_odderon_distribution_r_to_csv(10.01, 13, 36; μ=0, solver=:vegas)
write_odderon_distribution_r_to_csv(13.01, 16, 36; μ=0, solver=:vegas)
write_odderon_distribution_r_to_csv(16.01, 19, 36; μ=0, solver=:vegas)
write_odderon_distribution_r_to_csv(19.01, 20, 10; μ=0, solver=:vegas)

# write_odderon_distribution_to_csv(1e-5, 0.9e-2, 36 ; μ=0, solver=:vegas) 
# write_odderon_distribution_to_csv(1e-2, 2, 200 ; μ=0, solver=:vegas)

# write_ft_cubic_color_corellator_to_csv(1,1, 1e-4, 3.001, 30; solver=:vegas)
# write_ft_cubic_color_corellator_to_csv(1,-1, 1e-4, 3.001, 30; solver=:vegas)
println(Dates.now(), " Finished writers.")

# ======================
