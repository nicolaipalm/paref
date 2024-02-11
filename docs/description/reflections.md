# Pareto reflections

**What are Pareto reflections?**
On a top level, Pareto reflections are functions which reflect certain properties of Pareto points.

**Why do we use Pareto reflections?**
We use Pareto reflections in order to *inject* certain properties of found Pareto points into *any* MOO algorithm.

**How do we use Pareto reflections in Paref?**
This is done by calling the ``apply_to_sequence`` method of some MOO algorithm (implemented in ``ParefMOO`` interface)
to the Pareto reflection.

All reflections are found within the ``paref.pareto_reflections`` module.
You can access the Pareto points fitting the properties best by calling the ``best_fits`` attribute.
Currently, Paref includes implementations of the following Pareto reflections
(illustrated by their corresponding property):

|                        Property                         |                                     Graphic                                      |        Pareto reflection         | Supported target space dimension |                                                             Note                                                             |
|:-------------------------------------------------------:|:--------------------------------------------------------------------------------:|:--------------------------------:|:--------------------------------:|:----------------------------------------------------------------------------------------------------------------------------:|
|                   Being an edge point                   |     ![Edge points](../graphics/plots/reflections/FindEdgePointsSequence.svg)     |        ``FindEdgePoint``         |               All                |                                                                                                                              |
|                      Filling a gap                      |             ![Fill gap](../graphics/plots/reflections/FillGap2D.svg)             |           ``FillGap``            |               All                |                           This reflection works best if the gap defining points are Pareto optimal                           |
| Having minimum (weighted) distance to some utopia point | ![Weighted norm](../graphics/plots/reflections/MinimizeWeightedNormToUtopia.svg) | ``MinimizeWeightedNormToUtopia`` |               All                |                                                                                                                              |
|           Being constrained to a defined area           |         ![Restricted](../graphics/plots/reflections/RestrictByPoint.svg)         |       ``RestrictByPoint``        |               All                | Be careful with this Pareto reflection. If the Restricting point is too restrictive the optimization is most likely to fail. |
|  Reflecting a user defined priority of the objectives   |         ![Priority](../graphics/plots/moo-algorithms/PrioritySearch.png)         |        ``PrioritySearch``        |               All                |                               This Pareto reflection requires an edge point of each component                                |
