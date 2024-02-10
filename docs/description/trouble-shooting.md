The good news:
Every of Paref's algorithms has underlying theoretical guarantees.
In other words, whenever the MOO does not perform as expected, there are only two possible reasons:
- a bug in the implementation of the blackbox function
- the optimization did not converge to the right value

# How to check if the blackbox function is implemented correctly?

Check again if the blackbox function's
- design space dimension
- target space dimension
- the bounds

are correct.
Next, check if the blackbox function's ``__call__`` method returns reasonable values by calling the
``bbf.y`` and ``bbf.x``attribute.
If you see any unexpected values, the blackbox function is most likely implemented incorrectly.

# How to check if the optimization did succeed?

The most common error in the optimization process results from a bad scaling of the target values.
Ensure that the target values are _approximately_ all in the same range, ideally between 0 and 1.
Do so by subtracting and multiplying all target values by appropriate **positive** constants
within the ``__call__`` methode of the blackbox function.

**⚠️WARNING⚠️**:
- the constant with which the target values are multiplied **must** be strictly positive. Otherwise, the Pareto front will get changed!
- **do not make this functional**, for example by calling a MinMaxScaler, as this would make the optimization
highly unstable.

<details>
<summary><b>Example</b></summary>
Assume the first value of some 2-dimensional bbf with method

```python
def __call__(self, x: np.ndarray) -> np.ndarray:
    ... # Some code implementing the relationship between x and f(x)
    return result
```

is known to be in the range of -100 to -1000
and the second value in the range of 0.001 to 0.002. This (rather extreme) setup would most likely cause a bad optimization.
Then, the following modification would be appropriate:


```python
def __call__(self, x: np.ndarray) -> np.ndarray:
    ... # Some code implementing the relationship between x and f(x)
    return (result-np.array([1000,-0.001])  # Subtract the lower bounds
           /np.array([900,0.001]))  # Divide by the range
```
</details>

It is most likely that the optimization will succeed after such an adjustment.

However, sometimes this is not enough, and we need a take a closer look at the optimization process.
Paref works as follows:
1. a GPR is trained on the evaluations
2. the (expensive) blackbox function is replaced by the (cheap) GPR
3. the concatination of the GPR with some Pareto reflection is minimized


<details>
<summary><b>Step 1.:</b> check if the training did succeed</summary>

This information is provided by the ``Info`` class:

```python
from paref.express.info import Info
info = Info(blackbox_function=bbf, training_iter=2000, learning_rate=0.05)  # Create an instance of the Paref Info class
Info.model_fitness
```

and have a look at the plot of the training. It should look like this:

[//]: # (# TODO image)

If you recognize the plot to be 'non-converging'
try **more training iterations** (e.g. ``training_iter=5000``) and check if the optimization did succeed.
```python
info = Info(blackbox_function=bbf, training_iter=5000, learning_rate=0.05)  # Create an instance of the Paref Info class
Info.model_fitness
```

If you recognize the plot to be non-convex (e.g. if there are any spikes) this is
mostly likely caused by a **bad scaling of the target values** (see above) or a learning rate too large.
For the latter, decrease the learning rate (e.g. ``learning_rate=0.01``) and check if the training succeeds.
You will mostly likely need to increase the number of training iterations (e.g. ``training_iter=5000``).
You can check if those changes did succeed by calling the ``Info`` class again.

```python
info = Info(blackbox_function=bbf, training_iter=5000, learning_rate=0.01)  # Create an instance of the Paref Info class
Info.model_fitness
```

After determining appropriate values for the learning rate and the number of training iterations, initialize
the MOO algorithm again with those values.
</details>

<details>
<summary><b>Step 2.:</b> check if the GPR is a good approximation of the blackbox function</summary>
How well the GPR approximates the blackbox function is determined by two factors:

- the complexity of the bbf
- the number of evaluations (i.e. training points)

The good news is that any GPR can approximate any bbf arbitrarily well if the number of evaluations is large enough.
In other words, the optimization will succeed as long as the target space is explored sufficiently.
The bad news is that there is no general rule of thumb for the number of evaluations.

However, Paref provides an indicator for the quality of the GPR by calling
```python
from paref.express.info import Info
info = Info(blackbox_function=bbf, training_iter=2000, learning_rate=0.05)  # Create an instance of the Paref Info class
Info.model_fitness
```
and having a look at the 'average uncertainty'. A high uncertainty suggests a bad approximation.
In such case and after ensuring all of the above steps,
you will need to explore the target space further.
Currently, you can only do this by
- calling the ``bbf.perform_lhc`` method again or
- manually adding new points to the target space by calling the ``bbf`` to some design.

We are working on additional methods for (automatically and optimally) exploring the target space.
If nothing of the above worked feel free to contact me!
</details>
