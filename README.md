[![Python Test & Lint](https://github.com/nicolaipalm/paref/actions/workflows/python-test.yml/badge.svg)](https://github.com/nicolaipalm/paref/actions/workflows/python-test.yml)  
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg?style=plastic)](https://www.python.org/downloads/)
# Paref

With Paref you can build and use problem tailored multi-objective optimization algorithms.

A multi-objective optimization problem comes with an idea of what property identified (Pareto) points should contain.
Why not construct a multi-objective optimization algorithm which searches precisely for those points?
With the language of PAreto REFlections this is now possible.
Paref provides a generic interface for general constructions of multi-objective optimization algorithms (see picture) based on Pareto reflections.

Basic usage:
- paref additionally offers a multitude of ready-to-use multi-objective optimization algorithms
- each algorithm attempts to return Pareto points with specific properties and is based on mathematical proofs
- based on your individual preference which Pareto points you are looking for choose the one which fits your task best
- Convince yourself of the properties by running different multi-objective optimization algorithms in the example module.

[Advanced usage](theory in paper):
- implement all parts of a multi-objective optimization algorithm (i.e. Pareto reflections, sequences of such and optimizers) by yourself or (partly) use some of the already implemented instances
- join them to form your individual multi-objective optimization algorithm by using Parefs generic interface

Check out Parefs [documentation]() to learn about the individual properties of every MOO algorithm and Pareto reflection.

(add short animation of the search process to visualize the MOO algorithms properties)

## Check sheet for proper use of Surrogate based optimization

[] Are the components of input and output within the same scale? Action: Normalization of input and output data
[] Do you have enough training data (Rule of thumb: 3 dots per input dimension; Ex: 10 input dimensions and 30 samples)? Action: Extend initial (LH) sampling
[] Was the training successful? Action: inspect trainingsprocess if error converged; if not raise the number of training iteration (Rule of thumb: at least 2000 training iterations)

## Step-by-step guide

### Build your own algorithm
[Graphic workflow]

with coding example



### Apply an algorithm to a MOO problem
[Graphic workflow]

with coding example (every example in examples folder is built that way)
