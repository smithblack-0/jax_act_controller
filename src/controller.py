"""
Creation contains the builders and other mechanisms
needed to create an ACT instance. It also includes
some degree of error checking.
"""

from typing import Optional, List, Dict, Union, Any, Tuple
from dataclasses import dataclass
from src.states import ACTStates

import jax.tree_util
import textwrap
from jax import numpy as jnp
from src.types import PyTree

def _broadcast_from_dim_0(tensor: jnp.ndarray,
                          length: int
                          )->jnp.ndarray:
    while len(tensor.shape) < length:
        tensor = tensor[..., None]
    return tensor

def _weight_and_add(original: jnp.ndarray,
                   new: jnp.ndarray,
                   halting_probabilities: jnp.ndarray
                   )->jnp.ndarray:





class ACT_Controller:

    @staticmethod
    def cache_accumulator(state: ACTStates,
                          name: str,
                          item: PyTree)->ACTStates:
        """
        Cache a new accumulator away for when we perform
        the iteration step.

        :param state: The act state.
        :param name: The name of the accumulator to cache
        :param item: What to actually cache
        :return: A new ACTStates state, with the item cached.
        :except: ValueError, if accumulator referenced does not exist
        :except: RuntimeError, if accumulator was already set.
        """


        if name not in state.updates:
            raise ValueError(f"Accumulator with name {name} was never setup")
        if state.updates[name] != None:
            raise RuntimeError(f"Accumulator with name {name} was already set")
        update = state.updates.copy()
        update[name] = item
        return state.replace(updates=update)

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
            Original accumulator of shape {original.shape} does not match
            update accumulator of shape {new.shape}. This is not allowed
            """
            msg = textwrap.dedent(msg)
            raise RuntimeError(msg)

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
        # Logically, it should be noted that for error checking
        # to work we have to reset the updates back to none
        # during the iterate process.

        if halting_probabilities.shape != state.probabilities.shape:
            raise ValueError("provided and original shapes of halting probabilities do not match")

        # Unload the state values that will be changing. Make copies where appropriate to avoid
        # side effects

        iteration = state.iteration
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
        iteration += 1






        iteration = state.iteration
