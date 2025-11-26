module GellMann
using LinearAlgebra

export gell_mann, sun_symbols,
       dabc, fabc

"""
    gell_mann(n)

Return a vector of (n^2 - 1) SU(n) generators as matrices
with normalization tr (t^a t^b) = delta^{ab} / 2.
"""
function gell_mann(n::Int)
    gens = []

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

Compute f^{abc} and d^{abc} symbols for the given generators gens.

Returns (fabc,dabc) where:
    f^{abc} = -2i Tr(t^a [t^b, t^c])
    d^{abc} = 1/2 Tr(t^a {t^b, t^c})
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
# Arguments
- `a::Integer`: First index
- `b::Integer`: Second index
- `c::Integer`: Third index
- `n::Integer=3`: Dimension of the SU(n) group (default is 3)

# Returns
- d^{abc} symbol as Float64
"""
function dabc(a::Integer, b::Integer, c::Integer; n::Integer=3)
    gens = gell_mann(n)
    _, d_symbols = sun_symbols(gens)
    return d_symbols[a,b,c]
end

"""
    fabc(a, b, c; n=3)
Compute the antisymmetric f^{abc} symbol for SU(n).
# Arguments
- `a::Integer`: First index
- `b::Integer`: Second index
- `c::Integer`: Third index
- `n::Integer=3`: Dimension of the SU(n) group (default is 3)

# Returns
- f^{abc} symbol as Float64
"""
function fabc(a::Integer, b::Integer, c::Integer; n::Integer=3)
    gens = gell_mann(n)
    f_symbols, _ = sun_symbols(gens)
    return f_symbols[a,b,c]
end

# End module
end