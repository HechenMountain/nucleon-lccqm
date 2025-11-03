# This code performs parallel computation of the electromagnetic form factors,
# odderon distribution and gluon Sivers function
# over a specified range of Δ and k values, using multiple processors.
# The results are stored in CSV files.
# Just uncomment the function calls at the bottom to use.
# Run for example over command line via: nohup julia -p nworkers writers.jl > writers.log 2>&1 &
# Where nworkers is the number of parallel workers to use.

# ======================
# Imports
# ======================

using Distributed
using SharedArrays
using Dates

# Load the module locally first
@everywhere const CorePath = joinpath(@__DIR__,"core.jl")
include(CorePath)
using .Sivers

# Make it available to all workers
println("Finished module import at top level ", Dates.now())
println("Starting module import for $(nworkers()) workers ", Dates.now())
flush(stdout)

@everywhere begin
    # Import module if not already imported
    if !isdefined(Main, :Sivers)
        include(CorePath)
        using .Sivers
    end
    using SharedArrays
end

println("Finished module import at ", Dates.now())
flush(stdout)

# ======================
# Writers
# ======================

"""
        write_cuba_to_file(filename::String, results::SharedMatrix{Float64})

Write results from a computation to a CSV file.

# Arguments
- `filename`: The name of the output CSV file
- `results`: A shared matrix containing the computation results. Each row contains:
             [k, val_re, val_im, err_re, err_im, prob_re, prob_im, neval, fail, nregions]

# Returns
Nothing. Creates a CSV file with the specified filename containing the results.
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
end

"""
    write_2d_cuba_to_file(filename::String, k_list::AbstractVector{<:AbstractVector}, results::AbstractMatrix{Float64})

Write a CSV file for 2D k-grid results with columns: k_x,k_y,results

- `k_list` should be an array of [kx,ky] pairs (one per row of `results`).
- `results` is a matrix with numeric fields for each k (it may include the k magnitude in col 1
  if present). The writer will stringify columns 1..end of `results` for each row into a single
  bracketed, comma-separated `results` field in the CSV.
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
    write_gluon_sivers_to_csv(kmin::Real, kstep::Real, kmax::Real, μ::Real, [solver]::String="cuhre")
Write result of gluon_sivers for |k| in [kmin,kmax] GeV in steps of kstep GeV.
    
# Arguments
- `kmin`: Minimum value of k in GeV
- `kstep`: Step interval for k in GeV
- `kmax`: Maximum value of k in GeV
- `μ`: Regulator for integration
- `solver`: Integration strategy. Options: "cuhre" (default), "vegas", "divonne", "suave"

# Returns
Nothing. Creates a CSV file with the specified filename containing the results.
"""
function write_gluon_sivers_to_csv(kmin::Real, kstep::Real, kmax::Real,
                                   μ::Real, solver::String="cuhre")
    k_list = collect(kmin:kstep:kmax)
    n = length(k_list)
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
    filename = "output_gluon_sivers_$(solver)_$(μ)_$(kmin)_$(kmax).csv"
    write_cuba_to_file(filename,results)
end

@everywhere function test(k)
    integral, err, prob, neval, fail, nregions = [exp(-k[1]^2 - k[2]^2),exp(-k[1]^2 - k[2]^2)], [0.0,0.0], [1.0,1.0], 0, 0, 0
    return integral, err, prob, neval, fail, nregions
end

"""
    write_2d_odderon_distribution_to_csv(s01::Integer,s02::Integer,kmin::Real, kstep::Real, kmax::Real, μ::Real, [solver]::String="cuhre")

Write result of odderon_distribution for k_x,k_y in [-kmax,-kmin] ∪ [kmin,kmax] GeV in steps of kstep GeV.

# Arguments
- `s01, s02`: Spins of the ingoing/outgoing protons (each must be either +1 or -1)
- `kmin`: Minimum value of k in GeV
- `kstep`: Step interval for k in GeV
- `kmax`: Maximum value of k in GeV
- `μ`: Regulator for integration
- `solver`: Integration strategy. Options: "cuhre" (default), "vegas", "divonne", "suave"

# Returns
Nothing. Creates a CSV file with the specified filename containing the results.
"""
function write_2d_odderon_distribution_to_csv(s01::Integer,s02::Integer,kmin::Real, kstep::Real, kmax::Real,
                                              μ::Real, solver::String="cuhre")
    # Build 1D k values excluding the central region [-kmin, kmin].
    # We take negative side from -kmax to -kmin and positive side from kmin to kmax.
    neg_vals = collect(-kmax:kstep:-kmin)
    pos_vals = collect(kmin:kstep:kmax)
    k_vals = vcat(neg_vals, pos_vals)
    # Create 2D list of [kx, ky] pairs covering the Cartesian product of k_vals.
    k_list = [[kx, ky] for kx in k_vals for ky in k_vals]
    n = length(k_list)
    # columns: k, val_re, val_im, err_re, err_im, prob_re, prob_im, neval, fail, nregions
    results = SharedArray{Float64}(n,10)

    @sync @distributed for i in 1:n
        k = k_list[i]
        # integral, err, prob, neval, fail, nregions = test(k)
        integral, err, prob, neval, fail, nregions = Sivers.odderon_distribution(s01, s02, k, [0,0]; μ=μ, solver=solver)
        # store magnitude |k| in the first column (existing CSV expects a scalar k)
        k_mag = hypot(k[1], k[2])
        results[i, :] .= (Float64(k_mag), Float64(integral[1]), Float64(integral[2]), 
                          Float64(err[1]), Float64(err[2]), Float64(prob[1]), Float64(prob[2]),
                          Float64(neval), Float64(fail), Float64(nregions))
    end
    # Write to file
    filename = "output_2d_$(s01)_$(s02)_$(solver)_$(μ)_$(kmin)_$(kmax).csv"
    write_2d_cuba_to_file(filename,k_list,results)
end


"""
    write_odderon_distribution_to_csv(kmin::Real, kstep::Real, kmax::Real, μ::Real, [solver]::String="cuhre")

Write result of odderon_distribution for |k| in [kmin,kmax] GeV in steps of kstep GeV.

# Arguments
- `kmin`: Minimum value of k in GeV
- `kstep`: Step interval for k in GeV
- `kmax`: Maximum value of k in GeV
- `μ`: Regulator for integration
- `solver`: Integration strategy. Options: "cuhre" (default), "vegas", "divonne", "suave"

# Returns
Nothing. Creates a CSV file with the specified filename containing the results.
"""
function write_odderon_distribution_to_csv(kmin::Real, kstep::Real, kmax::Real,
                                           μ::Real, solver::String="cuhre")
    k_list = collect(kmin:kstep:kmax)
    n = length(k_list)
    # columns: k, val_re, val_im, err_re, err_im, prob_re, prob_im, neval, fail, nregions
    results = SharedArray{Float64}(n,10)

    @sync @distributed for i in 1:n
        k = k_list[i]
        integral, err, prob, neval, fail, nregions = Sivers.odderon_distribution(1, -1, [k,0], [0,0]; μ=μ, solver=solver)
        results[i, :] .= (Float64(k), Float64(integral[1]), Float64(integral[2]), 
                          Float64(err[1]), Float64(err[2]), Float64(prob[1]), Float64(prob[2]),
                          Float64(neval), Float64(fail), Float64(nregions))
    end
    # Write to file
    filename = "output_$(solver)_$(μ)_$(kmin)_$(kmax).csv"
    write_cuba_to_file(filename,results)
end

"""
    write_f1_form_factor_to_csv()

Write result of F1 form factor for |Δ| in [0,3.3] GeV

# Arguments
- `Δmin`: Minimum value of Δ in GeV
- `Δstep`: Step interval for Δ in GeV
- `Δmax`: Maximum value of Δ in GeV

# Returns
Nothing. Creates a CSV file with the specified filename containing the results.
"""
function write_f1_form_factor_to_csv(Δmin::Real=0.0, Δstep::Real=0.125, Δmax::Real=3.3)
    Δ_list = collect(Δmin:Δstep:Δmax)
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
    filename = "output_f1.csv"
    write_cuba_to_file(filename,results)
end

"""
    write_f2_form_factor_to_csv(Δmin::Real=1e-6, Δstep::Real=0.125, Δmax::Real=3.3)

Write result of F2 form factor for |Δ| in [1e-6,3.3] GeV

# Arguments
- `Δmin`: Minimum value of Δ in GeV
- `Δstep`: Step interval for Δ in GeV
- `Δmax`: Maximum value of Δ in GeV

# Returns
Nothing. Creates a CSV file with the specified filename containing the results.
"""
function write_f2_form_factor_to_csv(Δmin::Real=1e-6, Δstep::Real=0.125, Δmax::Real=3.3)
    Δ_list = collect(Δmin:Δstep:Δmax)
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
    filename = "output_f2.csv"
    write_cuba_to_file(filename,results)
end

"""
    write_f_form_factor_to_csv(s01::Integer,s02::Integer,Δmin::Real=1e-6, Δstep::Real=0.125, Δmax::Real=3.3)

Write result of general F-type form factor for |Δ| in [1e-6,3.3] GeV

# Arguments
- `s01, s02`: Spin of ingoing proton (must be either +1 or -1)
- `Δmin`: Minimum value of Δ in GeV
- `Δstep`: Step interval for Δ in GeV
- `Δmax`: Maximum value of Δ in GeV

# Returns
Nothing. Creates a CSV file with the specified filename containing the results.
"""
function write_f_form_factor_to_csv(s01::Integer,s02::Integer,Δmin::Real=1e-6, Δstep::Real=0.125, Δmax::Real=3.3)
    Δ_list = collect(Δmin:Δstep:Δmax)
    n = length(Δ_list)
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
    filename = "output_f_$(s01)_$(s02).csv"
    write_cuba_to_file(filename,results)
end

# ======================
# Main calls
# ======================

println("Starting writers at ", Dates.now())
flush(stdout)
# Add calls to the functions here as desired
# write_f1_form_factor_to_csv()
# write_f2_form_factor_to_csv()
# write_f_form_factor_to_csv(1,-1)
# write_odderon_distribution_to_csv(1e-4, 0.1, 1.0001, 0.0, "vegas")
# write_gluon_sivers_to_csv(1e-4, 0.1, 1.0001, 0.0, "vegas")
# write_2d_odderon_distribution_to_csv(1,-1,1e-4, 0.05, 1.001, 0.0, "vegas")
# write_2d_odderon_distribution_to_csv(-1,1,1e-4, 0.05, 1.001, 0.0, "vegas")
write_2d_odderon_distribution_to_csv(1,-1,1e-4, 0.01, 0.25, 0.0, "vegas")
println("Finished writers at ", Dates.now())

# ======================
