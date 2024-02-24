[![Python Test & Lint](https://github.com/nicolaipalm/paref/actions/workflows/python-test.yml/badge.svg)](https://github.com/nicolaipalm/paref/actions/workflows/python-test.yml)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg?style=plastic)](https://www.python.org/downloads/)

[documentation](https://paref.readthedocs.io/en/latest/)//[notebooks](https://github.com/nicolaipalm/paref/tree/main/docs/notebooks)//[demo]()

# Paref - problem tailored MOO for expensive black-box functions

A multi-objective optimization (MOO) problem comes with an idea of what
properties identified (Pareto) points must satisfy.
The fact that these characteristics are fulfilled is what makes a MOO successful in the first place.
Why not construct a MOO algorithm which searches exactly for those properties?
With the language of PAreto REFlections this is now possible.
This package contains...

- a series of ready-to-use [MOO algorithms](https://github.com/nicolaipalm/paref/tree/main/paref/moo_algorithms)
  corresponding to frequently targeted properties
- a framework for you to implement your problem tailored MOO algorithm
- generic and intuitive [interfaces](https://github.com/nicolaipalm/paref/tree/main/paref/interfaces) for MOO
  algorithms, black-box functions and more, so solving a MOO problem with user-defined properties with Paref requires
  only minimal effort

See the official [documentation](https://paref.readthedocs.io/en/latest/) for more information.

The official release is available at PyPi:

```
pip install paref
```
