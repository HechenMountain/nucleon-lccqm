# sivers.jl

**A Julia module to compute the gluon Sivers function in a light-cone constituent quark model** 

## About
We parametrize the light-cone baryon wavefunction in terms of its truncated Fock space content. Truncating the Fock space at the minimum Fock sector of three quarks, we compute the electromagnetic Dirac and Pauli form factors and find excellent agreement with the data, and in particular the anomalous magnetic moment is found to be F_2(0) = 1.82. We then obtain the cubic color corellator and after integrating it over the eikonal momenta, the gluon Sivers function at non-vanishing transverse momentum transfer.

## 📦 Features
- Different parameter sets can be used and are specifed in parameters.jl

## 🛠 Installation
Download the repo and use the writers.jl script to generate the desired output utilizing parallelization.

Alternatively, include the paths to the module(s) in a Jupyter notebook. E.g.
- include("/home/.../core.jl")

and then in a Jupyter cell
- using .Sivers

## ⚙️ Configuration
Adjust model parameters in parameters.jl to try different parameter sets.

## 🚀 Example Usage
Take a look at sivers.ipynb for examples

