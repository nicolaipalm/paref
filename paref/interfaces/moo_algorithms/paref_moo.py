from abc import abstractmethod
from typing import Optional, List, Union

import numpy as np

from paref.black_box_functions.design_space.bounds import Bounds
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import \
    SequenceParetoReflections
from paref.interfaces.moo_algorithms.stopping_criteria import StoppingCriteria
from paref.pareto_reflections.operations.compose_sequences import ComposeSequences


class ParefMOO:
    """Generic interface for MOO algorithms in Paref

    Mathematically, an MOO algorithm is expressed as a sequence

    .. math::

        (\mathcal{A}_i)_{i\\in \mathbb{N}}

    where each

    .. math::

        \mathcal{A}_i

    can be applied to a
    :py:class:`blackbox function <paref.interfaces.moo_algorithms.blackbox_function.BlackboxFunction>`
    and returns (approximately) a subset of its pareto front.

    A Pareto reflection based MOO algorithm is the of the form
    .. math::

        (\mathcal{A}_i)_{i\\in \mathbb{N}}

    where each

    .. math::

        \mathcal{A}_i

    is applied to a composition

    .. math::

        p_i \\circ f

    where :math:`p_i` is a Pareto reflection and :math:`f` is the blackbox function.

    This class provides the functionality needed.
    In addition, it allows you to

    **construct new algorithm** from a sequence of Pareto reflections with minimal effort simply by implementing its
    :meth:`sequence of Pareto reflections property
    <paref.interfaces.moo_algorithms.paref_moo.ParefMOO.sequence_of_pareto_reflections>`

    **apply** an MOO algorithm to a blackbox function and a sequence of Pareto reflections with minimal effort simply by
    calling its ::meth:`apply to sequence method
    <paref.interfaces.moo_algorithms.paref_moo.ParefMOO.apply_to_sequence>`.

    Examples
    --------
    # TODO: implement algo, implement by sequence, apply to sequence


    """

    @abstractmethod
    def apply_moo_operation(self,
                            blackbox_function: BlackboxFunction,
                            ) -> None:
        """Apply the MOO operation to the blackbox function

        This is the core of any MOO algorithm. In the language from above, this is the assignment :math:`\mathcal{A}_i`.


        .. warning::

            Applying the MOO operation to the blackbox function must include:
            #. determining a (potential) Pareto point
            #. evaluating the blackbox function at that point

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            blackbox function to which the MOO operation is applied

        """
        raise NotImplementedError

    # TODO: how to supported domain? Or needed?
    @property
    @abstractmethod
    def supported_codomain_dimensions(self) -> Optional[List[int]]:
        # If None then all codomain dimensions are supported
        """Supported codomain dimensions

        This includes all the dimensions of the target space this MOO supports, i.e. to which this MOO can be applied.
        For example, if the MOO algorithm is a minimization algorithm, then, the only supported dimension is one, i.e.
        this property must return [2].

        .. warning::

            If all dimensions are supported, then, this property must return None!

        Returns
        -------
        Optional[List[int]]
            list of supported dimensions or None if all dimensions are supported

        """
        raise NotImplementedError

    def __call__(self,
                 blackbox_function: BlackboxFunction,
                 stopping_criteria: StoppingCriteria,
                 ) -> None:
        """Apply the algorithm to a blackbox function

        .. note::
            If the given stopping criteria is met, then, the algorithm terminates as well as if the end of a sequence
            of Pareto reflections is reached.

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            blackbox function to which the algorithm is applied

        stopping_criteria : StoppingCriteria
            stopping criteria at which the algorithm must terminate

        Returns
        -------

        """

        sequence_of_pareto_reflections = self.sequence_of_pareto_reflections

        if sequence_of_pareto_reflections is None:
            while not stopping_criteria(blackbox_function):
                # TODO: monitoring
                self.apply_moo_operation(blackbox_function)

        else:
            self.apply_to_sequence(blackbox_function, sequence_of_pareto_reflections, stopping_criteria)

    def apply_to_sequence(self,
                          blackbox_function: BlackboxFunction,
                          sequence_pareto_reflections: Union[SequenceParetoReflections, ParetoReflection],
                          stopping_criteria: StoppingCriteria,
                          ):
        moo_sequence_of_pareto_reflections = self.sequence_of_pareto_reflections
        if moo_sequence_of_pareto_reflections is not None:
            sequence_pareto_reflections = ComposeSequences(sequence_pareto_reflections,
                                                           moo_sequence_of_pareto_reflections)

        while not stopping_criteria(blackbox_function):
            if isinstance(sequence_pareto_reflections, SequenceParetoReflections):
                # compose: caution what if one returns None
                pareto_reflection = sequence_pareto_reflections.next(blackbox_function)
                if pareto_reflection is None:
                    print('End of sequence reached. Algorithm stopped.')
                    break
                else:
                    composition_function = CompositionWithParetoReflection(blackbox_function=blackbox_function,
                                                                           pareto_reflection=pareto_reflection)

            elif isinstance(sequence_pareto_reflections, ParetoReflection):
                composition_function = CompositionWithParetoReflection(blackbox_function=blackbox_function,
                                                                       pareto_reflection=sequence_pareto_reflections)

            else:
                raise ValueError(
                    'sequence_pareto_reflections must be an instance of sequence '
                    'of Pareto reflections or a single Pareto reflection!')
            self.apply_moo_operation(composition_function)

    @property
    def sequence_of_pareto_reflections(self) -> Union[SequenceParetoReflections, ParetoReflection, None]:
        return None


class CompositionWithParetoReflection(BlackboxFunction):
    def __init__(self, blackbox_function: BlackboxFunction, pareto_reflection: ParetoReflection):
        if blackbox_function.dimension_target_space != pareto_reflection.dimension_domain:
            raise ValueError(
                f'Dimension of target space ({blackbox_function.dimension_target_space}) of blackbox function and '
                f'domain ({pareto_reflection.dimension_domain}) of Pareto reflection must match!')
        self._blackbox_function = blackbox_function
        self._pareto_reflection = pareto_reflection
        self._evaluations = blackbox_function._evaluations

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._pareto_reflection(self._blackbox_function(x))

    @property
    def dimension_design_space(self) -> int:
        return self._blackbox_function.dimension_design_space

    @property
    @abstractmethod
    def dimension_target_space(self) -> int:
        return self._pareto_reflection.dimension_codomain

    @property
    @abstractmethod
    def design_space(self) -> Union[Bounds]:
        return self._blackbox_function.design_space
