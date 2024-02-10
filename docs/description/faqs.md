### Apply a MOO algorithm to a MOO problem

<details>
  <summary>What is the purpose of applying an MOO algorithm to a MOO problem?</summary>

In MOO problems we commonly consider a [blackbox function](https://en.wikipedia.org/wiki/Black_box) with multiple conflicting
targets.
We are interested in its [Pareto front](https://en.wikipedia.org/wiki/Pareto_front).
Since the function of interest is a *blackbox* function, determining the whole Pareto front
is not possible in general.
Instead, we apply an MOO algorithm in order to iteratively determine a finite subset of the Pareto front
of the blackbox function.
In addition, the outcome of this optimization is commonly evaluated against the desired [properties]() of the
determined Pareto points. This is highly problem and user specific.
Paref strives to provide algorithms which are mathematically proven to (approximately)
determine Pareto points with those properties.

</details>

<details>
  <summary>How do we apply an MOO algorithm in Paref?</summary>

Applying an MOO algorithm in Paref requires some steps:

#### Define design and target space

  > This consists of answering the following question:
  > - what are my design parameters? How many do I have?
  > - what values of those design parameters do I accept?
  > - what are my target parameters? Which of them do I want to include?



#### Define desired properties

  > This could be answering the following questions:
  > - Do I want Pareto points in a certain interval?
  > - Do I want the Pareto points to be evenly distributed?
  > - Do I want as many Pareto points as possible or am I only interested in a single Pareto point?
  > - Do I want as less evaluations of the blackbox function as possible (i.e. if the blackbox function is expensive to evaluate)?



#### Initialize corresponding MOO algorithm

  > See the tutorials for more information.



#### Implement and initialize blackbox function

  > See the tutorials for more information.



#### Apply problem tailored MOO algorithm to blackbox function

  > In order to apply some MOO algorithm simply call the algorithm to the blackbox function and some stopping criterion
  > (indicating when the algorithm should terminate).
  > The evaluations can then be accessed within the blackbox function ``blackbox_function.evaluations`` or
  > ``blackbox_function.x`` (for the input values) or ``blackbox_function.y`` (for the output values) property.

<details>
  <summary>Example</summary>

   0. We use a mathematical test function with three input dimensions all between zero and one (i.e. design space is given by three-dimensional unit cube) and with two output dimensions (i.e. target space is the real plane)
   1. We want to have an idea of the "dimension" of the Pareto front (i.e. the Pareto points representing the minima in
      components) with minimum number of evaluations
   2. Accordingly, we choose the ``FindEdgePoints`` algorithm:

  ```python
  from paref.moo_algorithms.multi_dimensional.find_edge_points import FindEdgePoints
  moo = FindEdgePoints()
  ```

3. We implement and initialize the blackbox function in Parefs' blackbox function interface

```python
import numpy as np
from paref.blackbox_functions.design_space.bounds import Bounds
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction


class TestFunction(BlackboxFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = np.array([np.sum(x ** 2), x[0]])
        return y

    @property
    def dimension_design_space(self) -> int:
        return 3

    @property
    def dimension_target_space(self) -> int:
        return 2

    @property
    def design_space(self) -> Bounds:
        return Bounds(upper_bounds=np.ones(self.dimension_design_space),
                      lower_bounds=np.zeros(self.dimension_design_space))


blackbox_function = TestFunction()
```

4. We apply the MOO algorithm to the blackbox function with a maximum number of five iterations and print the so found Pareto front:
```python
from paref.moo_algorithms.stopping_criteria.max_iterations_reached import MaxIterationsReached
moo(blackbox_function = blackbox_function,
    stopping_criteria = MaxIterationsReached(max_iterations=5))
print(f"Calculated Pareto front: {blackbox_function.pareto_front}")
```
</details>

</details>

<details>
  <summary>Why do we apply an MOO algorithm that way?</summary>

Applying an MOO algorithm to an MOO problem should be as generic as possible.
Accordingly, we clearly distinguish between the steps outlined above.
In particular, this allows us
- to provide a generic workflow
- to require as few parameters as possible for applying an instance of an MOO algorithm to a blackbox function (leading to more robust algorithms)
- greatly simplify the process of implementing a MOO algorithm

</details>



### Implement your blackbox function

<details>
  <summary>What is a blackbox function?</summary>

The blackbox function expresses the relationship between your design parameters and
your target features.
This relationship is mostly not known (thus, the term "blackbox") but can be
observed at a finite set of points.
In the context of MOO, there mostly are more than one target feature and those features
are conflicting.
Accordingly, the blackbox function consists of the following information
- the design parameters and their accepted values
- the target features, in particular the number of features
- an evaluation rule for assigning some fixed vector of design parameters its vector of target features

</details>

<details>
  <summary>How to implement a blackbox function in Paref?</summary>

Implementing a blackbox function in Paref is given by implementing the ``BlackboxFunction`` interface.
This requires implementing the above information:
- the dimension of the design space within the ``dimension_design_space(self) -> int`` property
- the accepted design values within the ``def design_space(self) -> Union[Bounds]`` property
- the dimension of the target space within the ``def dimension_target_space(self) -> int`` property
- the assignment of some vector of design parameters to its corresponding target features within the `` def __call__(self, x: np.ndarray) -> np.ndarray`` method

**Example:**
Lets consider a blackbox function which has three design parameters each in a range of zero and one with two target features.
For simplicity, the assignment is given by simply forgetting the third design parameter.
Accordingly, we are given the following information
- the dimension of the design space is 3
- the accepted design values are all values between zero and one for each design parameter
- the dimension of the target space is 2
- the assignment is given by forgetting the third design parameter

```python
import numpy as np
from paref.blackbox_functions.design_space.bounds import Bounds
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction


class TestFunction(BlackboxFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x[:1]

    @property
    def dimension_design_space(self) -> int:
        return 3

    @property
    def dimension_target_space(self) -> int:
        return 2

    @property
    def design_space(self) -> Bounds:
        return Bounds(upper_bounds=np.ones(self.dimension_design_space),
                      lower_bounds=-np.zeros(self.dimension_design_space))
```
</details>

<details>
  <summary>Why do we implement a blackbox function this way?</summary>

Implementing a blackbox function should be
- intuitive
- robust
- done with minimal effort

Implementing a blackbox function into a Parefs interface allows us to
- reduce the effort to a minimum
- include generic error checking and handling
- check that all needed information of the blackbox function is handed
- include functionality such as calculating the Pareto front, saving and loading

 </details>



### Construct a problem tailored MOO algorithm
<details>
  <summary>What is a Problem tailored MOO algorithm?</summary>

A problem tailored MOO algorithm is a MOO algorithm tailored to your individual expectation on the outcome
of the algorithm.
Paref focuses on the properties of Pareto points you target and provides you with implementations and construction
rules of algorithms which target those properties.

</details>

<details>
  <summary>How to construct a problem tailored MOO algorithm?</summary>

Constructing a problem tailored MOO algorithm is based on the concept of Pareto reflections.
Accordingly, in order to construct a problem tailored MOO algorithm, you need to specify the (sequence of)
Pareto reflection(s) which reflect your targeted properties.
After initializing the sequence/reflection you apply an existing MOO to a blackbox function and the sequence using the
``apply_to_sequence`` method of that
MOO algorithm.

</details>

### Build your own algorithm
<details>
  <summary>What is a Paref MOO algorithm?</summary>

  We consider an MOO algorithm as an iterative algirthmic search for Pareto points of some blackbox function.
Paref MOOs in addition incorporate the concept of constructing a MOO algorithm from a (sequence of) Pareto reflections in order to tailor MOO algorithms
to user defined requirements.
 </details>

<details>
  <summary>How to implement an MOO algorithm in Paref?</summary>

Constructing a classical MOO algorithm in Paref is given by implementing the ``ParefMOO`` interface.
This requires implementing the following properties/methods:

- ``def apply_moo_operation(self,
                            blackbox_function: BlackboxFunction,
                            ) -> None``
- ``def supported_codomain_dimensions(self) -> Optional[List[int]]``


Constructing a new MOO from a (sequence of) Pareto reflections is given by implementing
the
``
def sequence_of_pareto_reflections(self) -> Union[SequenceParetoReflections, ParetoReflection, None]``
property of an already existing MOO algorithm.
All of the underlying functionality is already taken care of by the ``ParefMOO`` interface.

**Example**



 </details>

<details>
  <summary>Why do we construct it this way?</summary>

Paref MOOs are designed to be

user friendly
intuitive
functionality under the hood
 </details>
