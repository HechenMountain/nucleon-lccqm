# sivers.jl

**A julia module to compute the gluon Sivers function at large x** 
## About
This code is intended for use in a jupyter notebook but it should also work in a julia repl. We parametrize the baryon wavefunction in terms of its valence quark content, which is a valid approximation at large parton x. The cubic color corellator is explicitly computed and evaluated for specific values of the momentum transfer to obtained the gluon Sivers function.

## 📦 Features
- Large-x gluon Sivers function
- Different parameter sets can be used and are specifed in parameters.jl

## 🛠 Installation
Just download the repo and include the paths to the module(s) in a jupyter notebook. E.g.
- include("/home/.../sivers.jl")
- include("/home/.../GellMann.jl")

and then in a jupyter cell

- using .Sivers
- using .GellMann

## ⚙️ Configuration
Adjust model parameters in parameters.jl to try different parameter sets.

## 🚀 Example Usage
Take a look at sivers.ipynb for examples

