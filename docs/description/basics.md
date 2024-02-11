# The Basics

In a nutshell, [Paref's algorithms]() let you target certain user-defined
[properties of Pareto points](./faqs.md) making the MOO as
efficient and fast as possible.
This makes Paref ideal for optimizing expensive blackbox functions.

## What is a property of a Pareto point?

Properties are basically a-priori Pareto point rankings of the user.
For example, in most cases the user is only interested in
1. the Pareto points minimizing some component
2. the 'real trade-off' closest to the theoretical global optimum (i.e. the utopia point)

Evaluating other Pareto points would just waste valuable resources and
an [MOO algorithm targeting only those Pareto points]()
is desirable. With Paref you can do exactly that.


## Workflow

The workflow of Paref is simple:
1. [Implement a blackbox function]()
2. Explore the target space (by calling the blackbox function's ``perform_lhc(n)`` method)
3. [Apply a MOO algorithm reflecting your preference]()
4. [Analyze the output]()
5. Repeat steps 2-4 until satisfied

<details>
<summary><b>Full Example</b></summary>

```python
import numpy as np
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.blackbox_functions.design_space.bounds import Bounds
from paref.express.express_search import ExpressSearch
from paref.express.info import Info

# 1: Implement a blackbox function
class TestBlackboxFunction(BlackboxFunction):  # Implement the blackbox function interface
    def __call__(self, x) -> np.ndarray:
        return np.array([x[0],
                         x[0] ** 2 + x[1] ** 2])  # The blackbox function f relation of design (x) and target (f(x))

    @property
    def dimension_design_space(self) -> int:  # The dimension of the design space
        return 2

    @property
    def dimension_target_space(self) -> int:  # The dimension of the target space
        return 2

    @property
    def design_space(
            self) -> Bounds:  # The bounds of the design space (lower bounds, upper bounds) as instance of the Bounds class
        return Bounds(lower_bounds=[-1,-1], upper_bounds=[1,1])


bbf = TestBlackboxFunction()  # Initialize the blackbox function

# 2: Explore the target space
bbf.perform_lhc(n=20)  # exploration based on the Latin Hypercube Sampling

# 3: Apply a MOO algorithm reflecting your preference
moo = ExpressSearch(blackbox_function=bbf)  # Create an instance of the Paref Express class
moo.minimal_search(max_evaluations=5)  # perform the MOO algorithm

# 4: Analyze the output
print(f"Pareto front of bbf:\n {bbf.pareto_front}") # have a look at the Pareto front
info = Info(blackbox_function=bbf)  # Create an instance of the Paref Info class
info.topology # have a look at the topology of the Pareto front
info.suggestion_pareto_points

# 3: Apply a MOO algorithm reflecting your preference
moo.priority_search(max_evaluations=5, priority=[80, 20])  # perform the MOO algorithm

# 4: Analyze the output
print(f"Pareto point matching your preference best: \n {moo.priority_point}")
bbf.save('./deleteme.npy')  # save the evaluations
bbf.clear_evaluations()
print(f"Current evaluations: {bbf.evaluations}")
bbf.load('./deleteme.npy')  # load the evaluations
print(f"Loaded evaluations: {bbf.evaluations}")
```

</details>

## The blackbox function
In Paref a blackbox function (bbf) is implemented in the [``BlackboxFunction``]() interface.
This requires to implement the following four methods/properties:
- evaluating a design
```python
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (value of the blackbox function at x)
```
> **❗️NOTE❗:️** In order to make the optimization process as efficient as possible, the target values should be scaled to be approximately in the same range
> by multiplying (and subtracting) them by an appropriate **positive** constant. For example, if the
> first component of the target values is in the range of -100 to -1000 and the second component in the range of 0.001 to 0.002,
> this will mostly likely cause a bad optimization. Multiplying the first component by 0.001 and the second component by 1000
> will stabilize the optimization.
- the design space dimension
```python
    @property
    def dimension_design_space(self) -> int:
        return (dimension of the design space)
```
- the target space dimension
```python
    @property
    def dimension_target_space(self) -> int:
        return (dimension of the target space)
```
- the lower and upper bounds of the design space
```python
    @property
    def design_space(self) -> Bounds:
        return Bounds(lower_bounds=[lower_bound_component_1,...], upper_bounds=[upper_bound_component_1,...])
```

<details>
<summary><b>Example</b></summary>

```python
import numpy as np
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.blackbox_functions.design_space.bounds import Bounds

class TestBlackboxFunction(BlackboxFunction):  # Implement the blackbox function interface
    def __call__(self, x) -> np.ndarray:
        return np.array([x[0]+x[1],
                         x[0]+x[2]])  # The blackbox function f relation of design (x) and target (f(x))

    @property
    def dimension_design_space(self) -> int:  # The dimension of the design space
        return 3

    @property
    def dimension_target_space(self) -> int:  # The dimension of the target space
        return 2

    @property
    def design_space(
            self) -> Bounds:  # The bounds of the design space (lower bounds, upper bounds) as instance of the Bounds class
        return Bounds(lower_bounds=[-1,-1,-1], upper_bounds=[1,1,1])


bbf = TestBlackboxFunction()  # Initialize the blackbox function
```

</details>

Everything else is handled by Paref:
- the exploration of the target space: ``bbf.perform_lhc(n_samples)`` (20 samples as a rule of thumb)
- storing the evaluations of the blackbox function (accessed by ``bbf.evaluations``, ``bbf.x`` and ``bbf.y``)
- saving and loading evaluations (``bbf.save(path)`` and ``bbf.load(path)``)
- calculating the pareto front (``bbf.pareto_front``)



## How to apply one of Paref Express' MOO algorithms?

Applying one of Paref's MOO algorithms is done in a single line of code:
Simply call the respective algorithm by determining the maximum number of evaluations of the blackbox function

Using the example above, the following code first explores the design space granting 20 evaluations,
then searches for the Pareto points minimizing some component and lastly the 'real trade-off'
closest to the theoretical global optimum granting 5 evaluations in total:

```python
    from paref.express.express_search import ExpressSearch

    bbf.perform_lhc(n=20)  # exploration based on the Latin Hypercube Sampling
    moo = ExpressSearch(blackbox_function=bbf)  # Create an instance of the Paref Express class
    moo.minimal_search(max_evaluations=5)  # perform the MOO algorithm
```

_After_ performing the minimal search, we can apply a quite useful algorithm:
An MOO reflecting the users preference for certain objectives.
For example, minimizing component one could be more important than minimizing component two, in numbers 80% and 20%.
Applying Paref Express' priority search will target the trade-off which reflects that priority:

```python
    moo.priority_search(max_evaluations=5, priority=[80, 20])  # perform the MOO algorithm
```

Other algorithms of Paref Express are:
- a search for the minimum of a specific objective (``moo.search_for_minima``)
- a further search for a 'real trade-off' (``moo.search_for_best_real_trade_off``)


## How to analyze the output?

Paref Express' Info class provides a helpful tool to answer the following questions and guiding further MOO:
- are the target objectives conflicting?
- is there a specific trade-off which I should prefer (and look for)? If so, why?
- what is the dimension of the Pareto front (i.e. a point, a line, a plane etc.)?
- what is the "shape" of the Pareto front (i.e. convex, concave, linear)?
- what minimum value in each component can I expect?

```python
    from paref.express.info import Info
    info = Info(blackbox_function=bbf)
```

from here you can access the following information:
- topology: the shape of your Pareto front (``Info.topology``)
- a suggestion for Pareto points to evaluate, how and why (``Info.suggestion_pareto_points``)
- minima: the estimated minima of each component (``Info.minima``)
- model fitness: how well the model approximates the bbf, how to improve it and how certain its estimation is (``Info.model_fitness``)
