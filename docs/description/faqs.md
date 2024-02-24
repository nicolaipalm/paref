#### FAQ's

<details>
  <summary>What is the purpose of applying an MOO algorithm to a MOO problem?</summary>

In MOO problems we commonly consider a [blackbox function](https://en.wikipedia.org/wiki/Black_box) with multiple conflicting
targets.
We are interested in its [Pareto front](https://en.wikipedia.org/wiki/Pareto_front) representing the
optimal trade-offs of the (conflicting) objectives.
Since the function of interest is a *blackbox* function, determining the whole Pareto front
is not possible in general.
Instead, we apply an MOO algorithm in order to iteratively determine a finite subset of the Pareto front
of the blackbox function.

</details>

<details>
  <summary>What are properties of Pareto points and why are they so important?</summary>
After each MOO, the user evaluates the output against his or her preference for certain Pareto points.
For example, the user may prioritise some objectives over others and accordingly prefer Pareto
points that reflect this preference. Any such preference is a property of the Pareto points.

Especially for expensive blackbox functions, the user wants to have as few evaluations of the blackbox function
as possible and at the same time the best possible Pareto points with respect to his preferences.
Evaluating other Pareto points would only waste valuable resources.
An MOO algorithm that explicitly targets only Pareto points with these characteristics is, by its nature,
inherently the best fit for the user's problem. This is what we call a problem tailored MOO algorithm and
what Paref provides.
</details>

<details>
  <summary>What algorithm should I choose for my problem?</summary>

As each algorithm targets different properties of the Pareto points,
you should choose the algorithm that best fits your preference.
As a starting point you could answer the following questions:

- Do you want as many Pareto points as possible or are you only interested in a (couple of) Pareto point(s)?
- Do you have a preference for some objectives?
- Do you first want to have a rough idea of the Pareto front?

Based on those answers you may have a look at [Paref's algorithms](./moo-algorithms.md)
and choose the one that fits your preference best.

A good initial algorithm is the ``ExpressSearch.minimal_search`` algorithm, which provides a rough idea of
the Pareto front. Then, the ``Info``class provides information about the Pareto front which help you guide the further
optimization process.

</details>

<details>
  <summary>How does Paref work?</summary>

We refer to [the basics](./basics.md) for the general workflow of Paref.
Paref's algorithm are all constructed as follows:

1. A [GPR](https://en.wikipedia.org/wiki/Gaussian_process_regression) is fitted to the evaluations of the blackbox function.
2. The (expensive) blackbox function is replaced by the (cheap) GPR.
3. The concatination of the GPR with some [Pareto reflection](./reflections.md) is optimized.

At the core, Pareto reflections are the mathematical concept that allows us to construct MOO algorithms targeting
certain properties of Pareto points.
See [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4668407) for
the mathematical background of Pareto reflections.

</details>

<details>
  <summary>Can I use a cyber-physical system (e.g. Matlab or Modelica model) as blackbox function?</summary>

Yes. Paref is well suited for this purpose, and was successfully applied in the past.
However, Paref does not currently provide a direct interface to these systems. You have to implement it yourself.
We are currently working on that issue.

</details>
