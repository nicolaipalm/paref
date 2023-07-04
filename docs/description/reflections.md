# Pareto reflections

**What are Pareto reflections?**
On a top level, Pareto reflections are functions which reflect certain properties of Pareto points.

**Why do we use Pareto reflections?**
We use Pareto reflections in order to *inject* certain properties of found Pareto points into *any* MOO algorithm.

**How do we use Pareto reflections in Paref?**
This is done by calling the ``apply_to_sequence`` method of some MOO algorithm (implemented in ``ParefMOO`` interface)
to the Pareto reflection.


Currently, Paref includes implementations of the following Pareto reflections
(illustrated by their corresponding property):

|                        Property                         |                                     Graphic                                      |        Pareto reflection         | Supported target space dimension |          Note          | Code |
|:-------------------------------------------------------:|:--------------------------------------------------------------------------------:|:--------------------------------:|:--------------------------------:|:----------------------:|:----:|
|                   Being an edge point                   |     ![Edge points](../graphics/plots/reflections/FindEdgePointsSequence.svg)     |        ``FindEdgePoint``         |               All                |||
|                      Filling a gap                      |             ![Fill gap](../graphics/plots/reflections/FillGap2D.svg)             |          ``FillGap2D``           |                2                 |||
| Having minimum (weighted) distance to some utopia point | ![Weighted norm](../graphics/plots/reflections/MinimizeWeightedNormToUtopia.svg) | ``MinimizeWeightedNormToUtopia`` |               All                |||
|           Being constrained to a defined area           |          ![Fill gap](../graphics/plots/reflections/RestrictByPoint.svg)          |       ``RestrictByPoint``        |               All                |||
