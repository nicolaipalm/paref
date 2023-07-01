## Sequences of Pareto reflections

**What are sequences of Pareto reflections?**
Sequences of Pareto reflections are a Pythonic implementation of a mathematical (possibly finite) sequence
of Pareto reflections.

**Why using sequences of Pareto reflections?**
With sequences of Pareto reflections we can target different (possibly excluding) properties of Pareto points
within one MOO run. For example, we have Pareto reflections which target a certain corner of the Pareto front.
By constructing a sequence of Pareto reflections using different Pareto reflections which target different corners
we can construct a sequence which targets all corners.

**How do we use sequences of Pareto reflections?**
This is done by calling the ``apply_to_sequence`` method of some MOO algorithm (implemented in the ``ParefMOO``
interface)
to the sequence of Pareto reflections.

Currently, Paref includes implementations of the following sequences of Pareto reflections
(illustrated by their corresponding property):

|                                     Property                                     |                                     Graphic                                     |              Sequence               | Supported target space dimension |       Note       | Code |
|:--------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------:|:-----------------------------------:|:--------------------------------:|:----------------:|:----:|
|                           Filling gaps of Pareto front                           | ![Fill Gaps](../graphics/plots/reflections/FillGapsOfParetoFrontSequence2D.svg) | ``FillGapsOfParetoFrontSequence2D`` |                2                 |||
|                    Being the edge points of the Pareto front                     |     ![Fill Gaps](../graphics/plots/reflections/FindEdgePointsSequence.svg)      |     ``FindEdgePointsSequence``      |               All                |||
|                Repeating a (list of) Pareto reflections (generic)                |                                                                                 |        ``RepeatingSequence``        |               All                | Generic sequence ||
| Repeating a single Pareto reflection until a stopping criterion is met (generic) |                                                                                 |   ``NextWhenStoppingCriteriaMet``   |               All                | Generic Sequence ||
