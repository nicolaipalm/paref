API Reference
=============
This page contains the API reference for public objects and functions in ``paref``.

.. autosummary::
    :toctree: api
    :recursive:

    paref.moo_algorithms
    paref.pareto_reflection_sequences
    paref.pareto_reflections
    paref.black_box_functions
    paref.interfaces
    paref.express


Paref Express
-------------
.. autosummary::

    paref.express.express_search
    paref.express.info


MOO algorithms
--------------
.. autosummary::

    paref.moo_algorithms.multi_dimensional.find_1_pareto_points
    paref.moo_algorithms.multi_dimensional.find_edge_points
    paref.moo_algorithms.two_dimensional.fill_gaps_of_pareto_front_2d
    paref.moo_algorithms.minimizer.gpr_minimizer



Stopping criteria
-----------------
.. autosummary::

    paref.moo_algorithms.stopping_criteria.convergence_reached
    paref.moo_algorithms.stopping_criteria.logical_or_stopping_criteria
    paref.moo_algorithms.stopping_criteria.max_iterations_reached


Sequences of Pareto reflections
-------------------------------
.. autosummary::

    paref.pareto_reflection_sequences.generic.next_when_stopping_criteria_met
    paref.pareto_reflection_sequences.generic.repeating_sequence
    paref.pareto_reflection_sequences.multi_dimensional.avoid_pareto_front
    paref.pareto_reflection_sequences.multi_dimensional.fill_gaps_of_pareto_front_sequence
    paref.pareto_reflection_sequences.multi_dimensional.find_1_pareto_points_for_all_components_sequence
    paref.pareto_reflection_sequences.multi_dimensional.find_edge_points_sequence
    paref.pareto_reflection_sequences.multi_dimensional.grid_search
    paref.pareto_reflection_sequences.two_dimensional.fill_gaps_of_pareto_front_sequence_2d

Pareto reflections
------------------
.. autosummary::

    paref.pareto_reflections.avoid_points
    paref.pareto_reflections.fill_gap
    paref.pareto_reflections.find_1_pareto_points
    paref.pareto_reflections.find_edge_points
    paref.pareto_reflections.find_maximal_pareto_point
    paref.pareto_reflections.minimize_g
    paref.pareto_reflections.minimize_weighted_norm_to_utopia
    paref.pareto_reflections.operations
    paref.pareto_reflections.priority_search
    paref.pareto_reflections.restrict_by_point


Interfaces
----------
.. autosummary::

    paref.interfaces.moo_algorithms.blackbox_function
    paref.interfaces.moo_algorithms.paref_moo
    paref.interfaces.moo_algorithms.stopping_criteria
    paref.interfaces.pareto_reflections.pareto_reflection
    paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections
