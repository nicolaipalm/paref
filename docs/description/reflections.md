# Pareto reflections

**What are Pareto reflections?**
On a top level, Pareto reflections are functions which reflect certain properties of Pareto points.

**Why do we use Pareto reflections?**
Pareto reflections allow us to construct MOO algorithms targeting certain properties of Pareto points.
In other words, the Pareto optimal solutions of some Pareto reflection corresponding to some properties are
those Pareto points that fit the properties best.

**How do we use Pareto reflections in Paref?**
This is done by calling the ``apply_to_sequence`` method of some MOO algorithm (implemented in ``ParefMOO`` interface)
to the Pareto reflection.

```python
from paref.moo_algorithms.minimizer.gpr_minimizer import GPRMinimizer
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
from paref.pareto_reflections.find_edge_points import FindEdgePoints

moo = GPRMinimizer()
reflection = FindEdgePoints(dimension=0,
                            blackbox_function=bbf)
stopping_criteria = MaxIterationsReached(max_iterations=1)
moo.apply_to_sequence(blackbox_function=bbf,
                      sequence_pareto_reflections=reflection,
                      stopping_criteria=stopping_criteria)
```

All reflections are found within the ``paref.pareto_reflections`` module.
You can access the Pareto points fitting the properties best by calling the ``best_fits`` attribute.
Currently, Paref includes implementations of the following Pareto reflections
(illustrated by their corresponding property):

|                               Property                               |                                     Graphic                                      |        Pareto reflection         | Supported target space dimension |                                                                                  Note                                                                                  |
|:--------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|:--------------------------------:|:--------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                         Being an edge point                          |     ![Edge points](../graphics/plots/reflections/FindEdgePointsSequence.svg)     |        ``FindEdgePoint``         |               All                |                                                                                                                                                                        |
| Being Pareto optimal among all points minimizing some function ``g`` |                                                                                  |     ``MinGParetoReflection``     |               All                | The points this Pareto reflection returns are n.n. Pareto optimal. They are if the Pareto points among the points minimizing ``g`` are Pareto optimal among all points |
|                            Filling a gap                             |             ![Fill gap](../graphics/plots/reflections/FillGap2D.svg)             |           ``FillGap``            |               All                |                                                This reflection works best if the gap defining points are Pareto optimal                                                |
|       Having minimum (weighted) distance to some utopia point        | ![Weighted norm](../graphics/plots/reflections/MinimizeWeightedNormToUtopia.svg) | ``MinimizeWeightedNormToUtopia`` |               All                |                                                                                                                                                                        |
|                 Being constrained to a defined area                  |         ![Restricted](../graphics/plots/reflections/RestrictByPoint.svg)         |       ``RestrictByPoint``        |               All                |                      Be careful with this Pareto reflection. If the Restricting point is too restrictive the optimization is most likely to fail.                      |
|         Reflecting a user defined priority of the objectives         |         ![Priority](../graphics/plots/moo-algorithms/PrioritySearch.png)         |        ``PrioritySearch``        |               All                |                                                    This Pareto reflection requires an edge point of each component                                                     |
