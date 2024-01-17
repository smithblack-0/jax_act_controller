"""
Creation contains the builders and other mechanisms
needed to create an ACT instance. It also includes
some degree of error checking.
"""

from typing import Optional, List, Dict, Union, Any, Tuple
from dataclasses import dataclass
from src.states import ACTStates
from src.immutable import Immutable
import jax.tree_util
import textwrap
from jax import numpy as jnp
from src.types import PyTree


class ACT_Controller(Immutable):
    """
    A controller for the ACT process.

    This is an ACT controller designed to be
    setup in the associated builder, and which will accumulate
    information across act steps for all targeted quantities.

    The class must be initialized with a ACTStates object, and
    will generally return a new instance of itself whenever it
    is modified.

    The controller is completely immutable. Once initialized, if a method
    producing a state change is called, a new controller is returned instead.

    ----- Fields ----
    probabilities: The cumulative probabilities for each batch element
    residuals: The residuals on each batch element
    iterations: The iteration each batch is on, or alternatively where
    accumulators: The accumulated results

    ----- Properties ----

    halt_threshold: Above this cumulative probability a accumulator is considered halted
    halted_batches: A bool tensor of batches that are halted. True means halted.
    is_completely_halted: If every batch element is halted, this is true
    is_any_halted: If at least one batch element is halted, this is true.

    ----- Methods -----





    """


    # Direct properties
    @property
    def probabilities(self)->jnp.ndarray:
        return self.state.probabilities

    @property
    def residuals(self)->jnp.ndarray:
        return self.state.residuals

    @property
    def iterations(self)->jnp.ndarray:
        return self.state.iteration

    @property
    def accumulators(self)->PyTree:
        return self.state.accumulators

    # Inferred properties
    @property
    def halt_threshold(self)->float:
        return 1-self.state.epsilon
    @property
    def halted_batches(self)->jnp.ndarray:
        return self.probabilities > self.halt_threshold

    @property
    def is_completely_halted(self)->bool:
        return jnp.all(self.halted_batches)

    @property
    def is_any_halted(self)->bool:
        return jnp.any(self.halted_batches)

    # Helper logic
    def _process_probabilities(self,
                               halting_probabilities: jnp.ndarray
                               )->Tuple[jnp.ndarray,
                                        jnp.ndarray,
                                        jnp.ndarray
                                        ]:
        """
        Process the halting probabilities into the clamped
        halting probabilities, the residuals, and the new cumulative probabilities
        :param halting_probabilities: The current halting probabilities
        :return: The clamped halting probabilities
        :return: The updated residuals
        :return: The updated cumulative probabilities
        """

        # We need to know what batches are just now halting
        # to capture that information in the residuals, and
        # what tensors will be halting to properly trim
        # the halting probabilities. Computations assume that
        # something that is going to be halted, but are not
        # currently halted, will halt in this iteration.

        will_be_halted = halting_probabilities + self.probabilities > self.halt_threshold
        newly_halting = will_be_halted & ~self.halted_batches

        # Master residuals are updated only the round that we
        # are newly halted. However, the unfiltered residuals
        # are still useful for probability clamping

        raw_residuals = 1 - halting_probabilities
        residuals = jnp.where(newly_halting, raw_residuals, self.residuals)

        # Halting probabilities are clamped to the raw residual entries
        # where they will be halted. This ensures total probability can
        # never exceed one.

        halting_probabilities = jnp.where(will_be_halted, halting_probabilities, raw_residuals)

        # Finally, cumulative probabilities are updated to contain the sum
        # of the halting probabilities and the current cumulative probabilities

        probabilities = self.probabilities + halting_probabilities

        return halting_probabilities, residuals, probabilities

    def _update_accumulator(self,
                            accumulator_value: jnp.ndarray,
                            update_value: Optional[jnp.ndarray],
                            halting_probabilities: jnp.ndarray
                            ) -> jnp.ndarray:
        """
        Performs an update step using an accumulator, halting
        probabilities, and the update.

        :param accumulator_value: The current accumulator
        :param update_value: The value to include in the update
        :param halting_probabilities: The halting probabilities
        :return: The new accumulator.
        """

        if update_value is None:
            msg = f"""
            The accumulator was never updated during this act iteration.

            It is impossible to proceed.
            """
            raise RuntimeError(msg)

        if accumulator_value.shape != update_value.shape:
            msg = f"""
            Original accumulator of shape {accumulator_value.shape} does not match
            update accumulator of shape {update_value.shape}. This is not allowed
            """
            msg = textwrap.dedent(msg)
            raise RuntimeError(msg)

        # We will need the halting probabilities and halting batches tensors
        # to be broadcastable, from the left, with the accumulator shapes.
        #
        # We will often need to unsqueeze on the last dimension to make this
        # happen. This section accomplishes this.
        halted_batches = self.halted_batches
        while len(halting_probabilities.shape) < len(accumulator_value.shape):
            halting_probabilities = halting_probabilities[..., None]
            halted_batches = halted_batches[..., None]

        # We weight then add. The result is then created into an update, but
        # only accumulators which have not already reached a halting state get updated.

        new_accumulators = accumulator_value + halting_probabilities * update_value
        return jnp.where(halted_batches, accumulator_value, new_accumulators)

    # Main loop logic
    def cache_accumulator(self,
                          name: str,
                          item: PyTree
                          )->'ACT_Controller':
        """
        Cache a new group of accumulator pytrees onto a configured
        entry. Caching will save the tree for the later weight-and-add process.

        :param name: The name of the accumulator to cache
        :param item: What to actually cache
        :except: ValueError, if accumulator referenced does not exist
        :except: RuntimeError, if accumulator was already set.
        :return: New ACT_Controller with updated state
        """

        state = self.state
        if name not in state.updates:
            raise ValueError(f"Accumulator with name {name} was never setup")
        if state.updates[name] != None:
            raise RuntimeError(f"Accumulator with name {name} was already set this iteration!")
        update = state.updates.copy()
        update[name] = item
        new_state = state.replace(updates=update)
        return ACT_Controller(new_state)
    def iterate_act(self,
                    halting_probabilities: jnp.ndarray
                    )->'ACT_Controller':
        """
        Iterates the act process. Returns the result

        :param halting_probabilities:
        :return: The act controller with the iteration performed.
        """
        if halting_probabilities.shape != self.probabilities.shape:
            raise ValueError("provided and original shapes of halting probabilities do not match")

        # Compute and manage probability quantities.
        #
        # This includes clamping the halting probabilities, and
        # making the new probabilities and residuals.

        halting_probabilities, probabilities, residuals = self._process_probabilities(halting_probabilities)

        # Perform updates on each accumulator. Fetch the
        # update, run the update method, store it.

        accumulators = self.accumulators.copy()
        updates = self.state.updates.copy()

        for name, accumulator_value in accumulators.items():
            try:
                update_value = updates[name]
                accumulators[name] = self._update_accumulator(accumulator_value,
                                                              update_value,
                                                              halting_probabilities)
                updates[name] = None # Resets update slot, so we do not think an update was already done.
            except Exception as err:
                msg = f"An error occurred regarding accumulator {name}: \n\n"
                raise RuntimeError(msg) from err

        # We update the iteration count only where we have
        # not reached the halting state

        new_iterations = self.iterations + 1
        iterations = jnp.where(self.halted_batches, self.iterations, new_iterations)

        # Create the updated state. Return.

        state = self.state.replace(
            iterations = iterations,
            residuals = residuals,
            probabilities=probabilities,
            accumulators=accumulators,
            updates = updates,
        )
        return ACT_Controller(state)

    def __init__(self, state: ACTStates):
        super().__init__()
        self.state = state
        self.lock()





