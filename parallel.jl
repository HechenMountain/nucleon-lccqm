# This code performs parallel computation of the odderon distribution
# over a specified range of k values, using multiple processors.
# The results are stored in a CSV file.
# Run for example over command line via: nohup julia parallel.jl > parallel.log 2>&1 &
using Distributed
using SharedArrays

# First add the workers with the project environment
addprocs(20)

# Load the module locally first
include("sivers.jl")
using .Sivers

# Now make it available to all workers
@everywhere begin
    # Load module on each worker
    Base.include(@__MODULE__, joinpath(@__DIR__, "sivers.jl"))
    using .Sivers
    using SharedArrays
end

# Define k range and other parameters
kmin, kstep, kmax = 1e-3, 1e-3, 21e-3
mu, solver = 0, "cuhre"
filename = "output_$(solver)_$(mu)_$(kmin)_$(kmax).csv"
k_list = collect(kmin:kstep:kmax)
n = length(k_list)

# Prepare SharedArray to store results
results = SharedArray{Float64}(n,10)  # k, val_re, val_im, err_re, err_im, prob_re, prob_im, neval

# Perform parallel computation and write to CSV
open(filename, "w") do io 
    println(io, "k,val_re,val_im,err_re,err_im,prob_re,prob_im,neval,fail,nregions")  # header
    @sync @distributed for k in kmin:kstep:kmax
        integral, err, prob, neval, fail, nregions  = odderon_distribution(1, -1, [k,0], [0,0]; mu=mu, solver=solver)
        results[Int((k - kmin)/kstep) + 1, :] = [k, integral[1], integral[2], err[1], err[2], prob[1], prob[2], neval, fail, nregions]
    end
    # Write results to CSV
    for i in 1:n
        k = results[i, 1]
        integral = results[i, 2:3]
        err = results[i, 4:5]
        prob = results[i, 6:7]
        neval = Int(results[i, 8])
        fail = Int(results[i, 9])
        nregions = Int(results[i, 10])
        println(io, "$(k),$(integral[1]),$(integral[2]),$(err[1]),$(err[2]),$(prob[1]),$(prob[2]),$(neval),$(fail),$(nregions)")
        flush(io) 
    end
end

println("Results written to $(filename)")

