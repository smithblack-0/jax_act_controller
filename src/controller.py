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

    # Direct properties
    @property
    def probabilities(self)->jnp.ndarray:
        return self.state.probabilities

    @property
    def residuals(self)->jnp.ndarray:
        return self.state.residuals

    @property
    def iterations(self)->int:
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

    # Helper logic

    def _process_probabilities(self,
                               halting_probabilities: jnp.ndarray

                               ):
    def _update_probabilities(cls,
                             state: ACTStates,
                             halting_probabilities: jnp.ndarray
                             )-> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        #TODO: Doublecheck and simplify.
        halt_threshold = cls.halt_threshold(state)
        current_probabilities = state.probabilities
        current_residuals = state.residuals

        will_be_halted = halting_probabilities + current_probabilities > halt_threshold
        currently_halted = current_probabilities > halt_threshold
        newly_halting = will_be_halted & ~currently_halted

        residuals = 1 - halting_probabilities
        residuals = jnp.where(newly_halting, residuals, 0)
        residuals = current_residuals + residuals

        halting_probabilities = jnp.where(will_be_halted, residuals, halting_probabilities)
        probabilities = current_probabilities + halting_probabilities

        return probabilities, halting_probabilities, residuals


    # Main logic.
    def cache_accumulator(self,
                          name: str,
                          item: PyTree
                          )->'ACT_Controller':
        """
        Cache a new group of accumulator pytrees onto a configured
        entry. Caching will save for the later weight-and-add process.

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
            raise RuntimeError(f"Accumulator with name {name} was already set")
        update = state.updates.copy()
        update[name] = item
        new_state = state.replace(updates=update)
        return ACT_Controller(new_state)
    @classmethod

    def __init__(self, state: ACTStates):
        super().__init__()
        self.state = state
        self.lock()

class ACT_Controller:
    @staticmethod
    def halt_threshold(state: ACTStates)->float:
        return 1 - state.epsilon



    @classmethod
    def _update_accumulator(cls,
                            accumulator_value: jnp.ndarray,
                            update_value: Optional[jnp.ndarray],
                            halting_probabilities: jnp.ndarray
                            )->jnp.ndarray:

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

        # Broadcast halting probabilities to match, then weight and add.
        while len(halting_probabilities.shape) < len(accumulator_value.shape):
            halting_probabilities = halting_probabilities[..., None]
        return accumulator_value + halting_probabilities*update_value

    @classmethod
    def iterate_act(cls,
                    state: ACTStates,
                    halting_probabilities: jnp.ndarray
                    )->ACTStates:
        """
        Iterates the act process. Returns the result

        :param state:
        :param halting_probabilities:
        :return:
        """
        if halting_probabilities.shape != state.probabilities.shape:
            raise ValueError("provided and original shapes of halting probabilities do not match")


        # Clamp the halting probabilities, and update the residuals
        # as is appropriate.

        probabilities, halting_probabilities, residuals = cls._update_probabilities(state, halting_probabilities)

        # Update the accumulators. Reset the update slots as we go.
        accumulators = state.accumulators.copy()
        updates = state.updates.copy()

        for name, accumulator_value in accumulators.items():
            try:

                update_value = updates[name]
                accumulators[name] = cls.update_accumulator()
                updates[name] = None # Resets update slot, so we do not think an update was already done.
            except Exception as err:
                msg = f"An error occurred on iteration {iteration} regarding accumulator {name}: \n\n"
                raise RuntimeError(msg) from err
