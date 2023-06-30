# Parefs' MOO algorithms

> 💡 Read [this]() in order to understand how to implement a MOO algorithm in Paref!

**What are Pareto reflections?**

**How do we use Pareto reflections?**

**Why do we use Pareto reflections?**

Currently, Paref includes implementations of the following MOO algorithms
(illustrated by their corresponding property):

|               Property               |                                          Graphic                                           |                         Algorithm                          | Supported target space dimension | Note | Code |
|:------------------------------------:|:------------------------------------------------------------------------------------------:|:----------------------------------------------------------:|:--------------------------------:|:----:|:----:|
|         Being an edge point          |            ![Edge points](../graphics/plots/moo-algorithms/FindEdgePoints.svg)             |                     [FindEdgePoints]()                     |    All    |||
| Filling the gaps of the Pareto front | ![Fill gaps of Pareto front](../graphics/plots/moo-algorithms/FillGapsOfParetoFront2D.svg) |                [FillGapsOfParetoFront2D]()                 |         2          |||
|       Being evenly distributed       |             ![Evenly Scanned](../graphics/plots/moo-algorithms/ScanEvenly.svg)             | [FindEdgePoints]() followed by [FillGapsOfParetoFront2D]() |         2          |||
