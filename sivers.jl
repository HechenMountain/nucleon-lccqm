module Sivers

using Base: pi
using Printf
using PyPlot

export  gaussian, plot_gaussian,
        gaussian_3d, plot_gaussian_3d

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