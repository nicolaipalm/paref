# Parefs' MOO algorithms

Paref provides a series of ready to use *generic* (mainly minimization algorithms)
and *problem tailored* (i.e. targeting certain properties) MOO algorithms implemented in the ``ParefMOO`` interface.

Currently, Paref includes implementations of the following MOO algorithms
(illustrated by their corresponding property):

|                             Property                              |                                          Graphic                                           |                         Algorithm                          | Supported target space dimension |          Note          | Code |
|:-----------------------------------------------------------------:|:------------------------------------------------------------------------------------------:|:----------------------------------------------------------:|:--------------------------------:|:----------------------:|:----:|
|                        Being an edge point                        |            ![Edge points](../graphics/plots/moo-algorithms/FindEdgePoints.svg)             |                     ``FindEdgePoints``                     |               All                |||
|               Filling the gaps of the Pareto front                | ![Fill gaps of Pareto front](../graphics/plots/moo-algorithms/FillGapsOfParetoFront2D.svg) |                ``FillGapsOfParetoFront2D``                 |                2                 |||
|                     Being evenly distributed                      |             ![Evenly Scanned](../graphics/plots/moo-algorithms/ScanEvenly.svg)             | ``FindEdgePoints`` followed by ``FillGapsOfParetoFront2D`` |                2                 |||
| Finding approximately a minimum with minimum number of iterations |                                                                                            |                      ``GPRMinimizer``                      |                1                 | Apply to expensive bbf ||
| Finding a minimum with high number of iterations (for cheap bbf)  |                                                                                            |            ``DifferentiallEvolutionMinimizer``             |                1                 |   Apply to cheap bbf   ||
