module GellMann
using LinearAlgebra

export gell_mann, sun_symbols,
       dabc, fabc

"""
    gell_mann(n)

Construct SU(n) generators normalized as tr(t^a t^b) = δ^{ab} / 2.

Arguments
- `n::Int`: SU(n) dimension

Returns
- `Vector{Matrix{ComplexF64}}`: generators ordered symmetric, antisymmetric, then diagonal

Notes
- Generators follow the physics normalization; canonical matrices are rescaled by 1/2
"""
function gell_mann(n::Int)
    gens = Matrix{ComplexF64}[]

    # Symmetric generators
    for k in 2:n
        for j in 1:k-1
            mat = zeros(ComplexF64, n, n)
            mat[j,k] = 1
            mat[k,j] = 1
            push!(gens, mat)
        end
    end

    # Antisymmetric generators
    for k in 2:n
        for j in 1:k-1
            mat = zeros(ComplexF64, n, n)
            mat[j,k] = -im
            mat[k,j] = +im
            push!(gens, mat)
        end
    end

    # Diagonal generators
    for l in 1:(n-1)
        mat = zeros(ComplexF64, n, n)
        norm = sqrt(2 / (l * (l + 1)))
        for j in 1:l
            mat[j,j] = 1
        end
        mat[l+1, l+1] = -l
        mat .*= norm
        push!(gens, mat)
    end

    # Normalize tr(t^a t^b) = δ^{ab} / 2 
    gens = [ (1/2)*T for T in gens ]

    return gens
end

"""
    sun_symbols(gens)

Compute SU(n) structure constants for Hermitian, traceless generators gens.

Arguments
- `gens::AbstractVector{<:AbstractMatrix}`: generators normalized with tr(t^a t^b) = δ^{ab} / 2

Returns
- `fabc::Array{Float64,3}`: antisymmetric structure constants from f^{abc} = -2im tr(t^a [t^b, t^c])
- `dabc::Array{Float64,3}`: symmetric structure constants from d^{abc} = 2 tr(t^a {t^b, t^c})

Notes
- Results are real for properly normalized Hermitian generators and assume length(gens) = n^2 - 1
"""
function sun_symbols(gens)
    dim = length(gens)
    fabc = zeros(Float64, dim, dim, dim)
    dabc = zeros(Float64, dim, dim, dim)

    for a in 1:dim
        for b in 1:dim
            for c in 1:dim
                comm = gens[b]*gens[c] - gens[c]*gens[b]
                anticomm = gens[b]*gens[c] + gens[c]*gens[b]
                # These relations are valid for tr(t^a t^b) = delta^{ab} / 2
                # and choice of anomaly coefficient A = 1 for SU(3)
                fabc[a,b,c] = -2im * (tr(gens[a]*comm))
                dabc[a,b,c] = 2 * tr(gens[a]*anticomm)
            end
        end
    end

    return fabc, dabc
end

"""
    dabc(a, b, c; n=3)
Compute the symmetric d^{abc} symbol for SU(n).

Arguments
- `a::Integer`: first index
- `b::Integer`: second index
- `c::Integer`: third index
- `n::Integer=3`: SU(n) dimension (default 3)

Returns
- `Float64`: d^{abc} symbol
"""
function dabc(a::Integer, b::Integer, c::Integer; n::Integer=3)
    gens = gell_mann(n)
    _, d_symbols = sun_symbols(gens)
    return d_symbols[a,b,c]
end

"""
    fabc(a, b, c; n=3)
Compute the antisymmetric f^{abc} symbol for SU(n).

Arguments
- `a::Integer`: first index
- `b::Integer`: second index
- `c::Integer`: third index
- `n::Integer=3`: SU(n) dimension (default 3)

Returns
- `Float64`: f^{abc} symbol
"""
function fabc(a::Integer, b::Integer, c::Integer; n::Integer=3)
    gens = gell_mann(n)
    f_symbols, _ = sun_symbols(gens)
    return f_symbols[a,b,c]
end

# ======================
end # module GellMann