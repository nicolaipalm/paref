from abc import abstractmethod
from typing import Optional, List, Union

import numpy as np

from paref.black_box_functions.design_space.bounds import Bounds
from paref.interfaces.moo_algorithms.blackbox_function import BlackboxFunction
from paref.interfaces.moo_algorithms.stopping_criteria import StoppingCriteria
from paref.interfaces.pareto_reflections.pareto_reflection import ParetoReflection
from paref.interfaces.sequences_pareto_reflections.sequence_pareto_reflections import \
    SequenceParetoReflections
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
    In particular, it allows you to

    **construct new algorithm** from a sequence of Pareto reflections simply by implementing its
    :meth:`sequence of Pareto reflections property
    <paref.interfaces.moo_algorithms.paref_moo.ParefMOO.sequence_of_pareto_reflections>`

    **apply** an MOO algorithm to a blackbox function and a sequence of Pareto reflections simply by
    calling its ::meth:`apply to sequence method
    <paref.interfaces.moo_algorithms.paref_moo.ParefMOO.apply_to_sequence>`.

    Examples
    --------
    # TBA: implement algo, implement by sequence, apply to sequence


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

    # TBA: how to supported domain? Or needed?
    @property
    @abstractmethod
    def supported_codomain_dimensions(self) -> Optional[List[int]]:
        # If None then all codomain dimensions are supported
        """Supported codomain dimensions

        This includes all the dimensions of the target space this MOO supports, i.e. to which this MOO can be applied.
        For example, if the MOO algorithm is a minimization algorithm, then, the only supported dimension is one, i.e.
        this property must return [2].

        .. warning::

            If all dimensions are supported, then, this property must return ``None``!

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
                # TBA: monitoring: value, if PP?
                number_evaluations = len(blackbox_function.evaluations)
                self.apply_moo_operation(blackbox_function)
                if len(blackbox_function.evaluations) == number_evaluations:
                    print('WARNING: algorithm did not evaluate or store the evaluation of the blackbox function!')

        else:
            self.apply_to_sequence(blackbox_function,
                                   sequence_of_pareto_reflections,
                                   stopping_criteria,
                                   with_underlying_sequence=False)

    def apply_to_sequence(self,
                          blackbox_function: BlackboxFunction,
                          sequence_pareto_reflections: Union[SequenceParetoReflections, ParetoReflection],
                          stopping_criteria: StoppingCriteria,
                          with_underlying_sequence: bool = True,
                          ):
        """Apply the algorithm the composition of a blackbox function with a (sequence of) Pareto reflection(s)

        Calling this method, applies to algorithm to the composition of the blackbox function with the pareto reflection
        (if a single Pareto reflection is provided) and with the next Pareto reflection obtained by the sequence (if a
        sequence of Pareto reflection is provided).

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            underlying blackbox function

        sequence_pareto_reflections : SequenceParetoReflections
            sequence or single Pareto reflection to compose with blackbox function

        stopping_criteria : StoppingCriteria
            indicator when the algorthm terminates

        with_underlying_sequence : bool default False
            decide whether sequence should be composed with implemented sequence of algorithm

        Returns
        -------

        """
        if with_underlying_sequence:
            moo_sequence_of_pareto_reflections = self.sequence_of_pareto_reflections

        else:
            moo_sequence_of_pareto_reflections = None

        if moo_sequence_of_pareto_reflections is not None:
            sequence_pareto_reflections = ComposeSequences(sequence_pareto_reflections,
                                                           moo_sequence_of_pareto_reflections)

        while not stopping_criteria(blackbox_function):
            number_evaluations = len(blackbox_function.evaluations)
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
            if len(blackbox_function.evaluations) == number_evaluations:
                print('WARNING: algorithm did not evaluate or store the evaluation of the blackbox function!')

    @property
    def sequence_of_pareto_reflections(self) -> Union[SequenceParetoReflections, ParetoReflection, None]:
        """Optional: Underlying sequence of MOO algorithm

        We can extend every MOO algorithm by applying it to a composition of the blackbox function and a Pareto
        reflection in each iteration.
        Implementing this property (i.e. a sequence of or a single Pareto reflection) will extent the underlying MOO
        algorithm in this fashion.

        .. note::

            This is an optional parameter.
            If a single Pareto reflection is implemented, then, the underlying MOO algorithm applied to some blackbox
            function will be applied to the composition of blackbox function with this Pareto reflection when called.
            This does not change the underlying :meth:`moo operation method
            <paref.interfaces.moo_algorithms.paref_moo.ParefMOO.apply_moo_operation>`!
            Similarly, if a sequence of reflections is implemented, then, the algorithm is applied to the composition
            with the next reflection obtained from the sequence.
            The sequence will be initialized *once* when the algorithm is called.

        Returns
        -------
        SequenceParetoReflections
            underlying sequence of Pareto reflection

        """
        return None


class CompositionWithParetoReflection(BlackboxFunction):
    """Wrapper for composing a Pareto reflection with a blackbox function in order to obtain a new blackbox function

    This class constructs a new blackbox function :math:`p\\circ f` out of a blackbox function :math:`f` and a Pareto
    reflection :math:`p`.
    In particular, its design space is given by the design space of the underlying blackbox function and its target
    space dimension is given by the dimension of the codomain dimension of the Pareto reflection.

    .. note::

        Calling this blackbox function will call the underlying blackbox function. In particular, this means that
        the result is stored in the underlying blackbox function.

    """

    def __init__(self, blackbox_function: BlackboxFunction, pareto_reflection: ParetoReflection):
        """Initilize blackbox function and Pareto reflection to be composed

        Parameters
        ----------
        blackbox_function : BlackboxFunction
            blackbox function

        pareto_reflection : ParetoReflection
            Pareto reflection
        """
        super().__init__()
        if blackbox_function.dimension_target_space != pareto_reflection.dimension_domain:
            raise ValueError(
                f'Dimension of target space ({blackbox_function.dimension_target_space}) of blackbox function and '
                f'domain ({pareto_reflection.dimension_domain}) of Pareto reflection must match!')
        self._blackbox_function = blackbox_function
        self._pareto_reflection = pareto_reflection
        self._evaluations = [[evaluation[0], pareto_reflection(evaluation[1])] for evaluation in
                             blackbox_function._evaluations]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the composition to an input

        I.e. evaluate the Pareto reflection at the evaluation of the blackbox function at the input x.

        Parameters
        ----------
        x : np.ndarray
            Input to which composition is applied

        Returns
        -------
        np.ndarray
            Output of composition applied to input

        """
        return self._pareto_reflection(self._blackbox_function(x))

    @property
    def dimension_design_space(self) -> int:
        """Dimension of design space

        I.e. dimension of design space of underlying blackbox function.

        Returns
        -------
        int
            dimension of design space of composition

        """
        return self._blackbox_function.dimension_design_space

    @property
    @abstractmethod
    def dimension_target_space(self) -> int:
        """Dimension of target space

        I.e. dimension of codomain of Pareto reflection.

        Returns
        -------
        int
            dimension of target space of composition

        """
        return self._pareto_reflection.dimension_codomain

    @property
    @abstractmethod
    def design_space(self) -> Union[Bounds]:
        """Design space of composition

        I.e. design space of the blackbox function.

        Returns
        -------
        Union[Bounds]
            pythonic representation of design space

        """
        return self._blackbox_function.design_space
