module Helpers

# Integration backends
using Cuba
using StaticArrays: SVector

# ======================
# Export
# ======================

export  δ,
        spin_index,
        sqnorm2,
        vec2,
        cuba_to_parton_x, 
        cuba_to_polar,
        cuba_to_hyperspherical,
        polar_to_cartesian,
        cartesian_to_polar,
        SOLVERS

# Mapping from solver symbols to Cuba backends
const SOLVERS = Dict{Symbol, Function}(
    :cuhre   => cuhre,
    :vegas   => vegas,
    :suave   => suave,
    :divonne => divonne,
)

# ======================
# Functions
# ======================

"""
    δ(a::Real, b::Real)

Kronecker delta.

Arguments
- `a`: first argument
- `b`: second argument

Returns
- `1` if `a == b`, otherwise `0`
"""
δ(a::Real, b::Real) = a == b ? 1 : 0

@inline sqnorm2(v::AbstractVector{<:Real}) = v[1] * v[1] + v[2] * v[2]
@inline vec2(x::Real, y::Real) = SVector(x, y)

"""
    spin_index(s::Integer)

Map spin values -1 and +1 to indices 1 and 2, respectively.

Arguments
- `s`: spin value (-1 or +1)

Returns
- `index::Integer`: index 1 for spin -1, index 2 for spin +1
"""
function spin_index(s::Integer)
    if s == -1
        return 1
    elseif s == 1
        return 2
    end
    throw(ArgumentError("Invalid spin value: $s. Expected -1 or 1."))
end

"""
    regulate_cuba(::Vector{<:Real})

Regulates the endpoints of [0,1]^n Cuba samples to avoid NaNs

Arguments
- `x`: [0,1]^n Cuba sample

Returns
- `x::Vector{<:Real}`: regulated values (x[i] +-= 1e-12 depending on the endpoint)
"""
function regulate_cuba(x::Vector{<:Real})
    if any(x .< 0) || any(x .> 1)
        throw(ArgumentError("Cuba value out of bounds [0,1]: $x"))
    end
    return clamp.(x, 1e-14, 1 - 1e-14)
end


"""
    cuba_to_parton_x(x::Vector{<:Real})

Performs the variable transformation from [0,1]^n 
Cuba samples to parton-x with the condition
1 - x[1] - ... - x[n] = 0 .

Arguments
- `x`: [0,1]^n Cuba sample

Returns
- `x::Vector{<:Real}`: parton-x values
- `jac::Real`: Jacobian determinant

Notes
- Assumes integration over Dirac delta has already been carried out
"""
function cuba_to_parton_x(x::Vector{<:Real})
    n = length(x) + 1 
    y = zeros(n)
    jac = 1.0
    # Compute recursively
    # y_1 = x_1
    # ...
    # y_{n-1} = (1 - sum_{k=1}^{n-2} y_k ) * x_{n-1}
    #         = prod_{k=1}^{n-2} (1-y_k) * x_{n-1}
    # y_n = 1 - sum_{k=1}^{n-1} y_k
    # jac = prod_{k=1}^{n-1} (1-y_k)
    rem = 1.0
    for i in 1:n-1
        y[i] = rem * x[i]
        jac *= rem
        rem -= y[i]
    end
    y[n] = rem
    return y, jac
end

"""
    cuba_to_polar(x::Vector{<:Real})

Transform a Cuba sample `x ∈ [0,1]^2` into polar coordinates.

Arguments
- `x`: sample point in the unit square `[0,1]^2`

Returns
- `r::Real`: radius
- `ϕ::Real`: azimuthal angle
- `jac::Real`: Jacobian determinant of the transformation

Notes
- Throws `ArgumentError` if called with `length(x) != 2`
"""
function cuba_to_polar(x::Vector{<:Real})
    n = length(x)            
    if n != 2
        throw(ArgumentError("Input must be two-dimensional for polar coordinates"))
    end
    # r ∈ [0, ∞)
    r = x[1] / (1 - x[1])
    dr = 1 / (1 - x[1])^2
    ϕ = 2π * x[2]
    dϕ = 2π
    jac = r * dr * dϕ

    return r, ϕ, jac
end

"""
    cuba_to_hyperspherical(x::Vector{<:Real})

Transform a Cuba sample `x ∈ [0,1]^n` into hyperspherical coordinates.

Arguments
- `x`: sample point in the unit hypercube `[0,1]^n` with `n ≥ 2`

Returns
- `r::Real`: radius
- `angles::Vector{Real}`: angular coordinates `[θ₁, θ₂, …, θ_{n-1}]`
- `jac::Real`: Jacobian determinant of the transformation

Notes
- Throws `ArgumentError` if called with `length(x) < 2`
"""
function cuba_to_hyperspherical(x::Vector{<:Real})
    n = length(x)            
    if n < 2
        throw(ArgumentError("Need at least 2 dimensions for hyperspherical coordinates"))
    end

    # r ∈ [0, ∞)
    r = x[1] / (1 - x[1])
    dr = 1 / (1 - x[1])^2

    thetas = zeros(eltype(x), n-1)

    jac = r^(n-1) * dr

    # first n-2 angles in [0, π]
    for i in 1:(n-2)
        cosθ = 2*x[i+1] - 1
        θ = acos(cosθ)
        thetas[i] = θ
        jac *= 2 * sin(θ)^(n-i-2)
    end

    # last angle in [0,2π)
    ϕ = 2π * x[n]
    thetas[end] = ϕ
    jac *= 2π

    return r, thetas, jac
end

"""
    polar_to_cartesian(vec_polar::Vector{<:Real})

Transform a polar input vector 'vec_polar=[v_r,v_ϕ]' 
to cartesian coordinates 'vec=[v_x,v_y]'

Arguments
- `vec_polar`: polar input vector

Returns
- `vec::Vector{<:Real}`: cartesian output vector

Notes
- Throws `ArgumentError` if called with `length(vec) != 2`
"""
function polar_to_cartesian(vec_polar::AbstractVector{<:Real})
    if length(vec_polar) != 2
        throw(ArgumentError("Polar to cartesian coordinates requires 2D input."))
    end
    r, ϕ = vec_polar
    x, y = r * cos(ϕ), r * sin(ϕ)
    return vec2(x, y)
end

"""
    cartesian_to_polar(vec::Vector{<:Real})

Transform a cartesian input vector 'vec=[v_x,v_y]' to polar coordinates
'vec_polar=[v_r,v_ϕ]'

Arguments
- `vec`: cartesian input vector

Returns
- `vec_polar::Vector{<:Real}`: polar output vector

Notes
- Throws `ArgumentError` if called with `length(vec) != 2`
"""
function cartesian_to_polar(vec::AbstractVector{<:Real})
    if length(vec) != 2
        throw(ArgumentError("Cartesian to polar coordinates requires 2D input."))
    end
    x, y = vec
    r = hypot(x, y)
    ϕ = atan(y, x)
    return vec2(r, ϕ)
end

# ======================
end # module Helpers