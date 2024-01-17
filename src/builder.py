"""
Creation contains the builders and other mechanisms
needed to create an ACT instance. It also includes
some degree of error checking.
"""

from typing import Optional, List, Union

import jax.tree_util
import textwrap
from jax import numpy as jnp

from src.states import ACTStates
from src.types import PyTree
from src.controller import ACT_Controller

class StateBuilder:
    """
    A builder to create an act state usable downstream. It works
    by configuring the batch dimensions across which to gather
    probabilities, then allows one to define any number of accumulator
    channels.

    Accumulator channels can
    This must build the ACT state, consisting of probabilities
    and residuals, output accumulators, and any error checking
    logic.
    """
    @staticmethod
    def _regularize_shape(shape: Union[int, List[int]])->List[int]:
        if isinstance(shape, int):
            shape = [shape]
        return shape

    def _validate_accumulator(self, name: str, array: jnp.ndarray):

        # Validate we are not setting more than once.
        if name in self.accumulators:
            raise ValueError(f"Accumulator of name {name} has already been setup")

        # Validate array dtype
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise ValueError(f"Attempt to setup accumulator of name {name} with a non floating dtype")

        # Validate batch length
        length = len(self.core_shape)
        if len(array.shape) < length:
            msg = f"""
            Accumulator named {name} with shape {array.shape} is shorter than 
            batch length of {length}
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)

        # Validate batch shape.
        # NOTE: coreshape is list, if you are having trouble here.
        if array.shape[:length] != self.core_shape:
            msg = f"""
            Accumulator with shape of shape {array.shape} does not posses
            batch dimensions matching {self.core_shape}
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)


    def setup_accumulator_by_shape(self,
                                   name: str,
                                   shape: Union[int, List[int]],
                                   dtype: Optional[jnp.dtype] = None,
                                   ):
        """
        Sets up a simple act accumulator by name and shape. It will
        be filled with zeros.

        You must provide the exact shape, including the batch shape.

        :param name: the name of the accumulator to setup
        :param shape: The shape. Must include the batch dimensions up front.
        :param dtype: The dtype of the setup accumulator
        """

        shape = self._regularize_shape(shape)
        accumulator = jnp.zeros(shape, dtype)
        self._validate_accumulator(name, accumulator)
        self.accumulators[name] = accumulator

    def setup_accumulator_like(self,
                               name: str,
                               pytree: PyTree,
                               ):
        """
        Sets up an accumulator like the provided
        pytree. It will be filled with zeros.

        :param name: The name of the accumulator
        :param pytree: The pytree to mime
        """

        pytree = jax.tree_util.tree_map(lambda x : jnp.zeros_like(x), pytree)
        jax.tree_util.tree_map(lambda x : self._validate_accumulator(name, x), pytree)
        self.accumulators[name] = pytree

    def setup_accumulator_by_pytree(self,
                                    name: str,
                                    pytree: PyTree):
        """
        Directly sets the default contents of an accumulator
        to be equal to a given pytree.

        :param name: The name of the accumulator
        :param pytree: The pytree to set the defaults to
        """

        jax.tree_util.tree_map(lambda x : self._validate_accumulator(name, x), pytree)
        self.accumulators[name] = pytree

    def override_probabilities(self, probabilities: jnp.ndarray):
        """
        Manually set the probabilities tensor to be something different
        :param probabilities: The probabilities tensor. Must be compatible with shape
        """
        if probabilities.shape != self.probabilities.shape:
            raise ValueError("Current and new probability tensors are not the same shape")
        self.probabilities = probabilities

    def override_residuals(self, residuals: jnp.ndarray):
        """
        Manually set the residuals tensors to be something of the
        same shape, but with different entries
        :param residuals: The residual tensor to set
        """
        if residuals.shape != self.residuals.shape:
            raise ValueError("Current and new residual tensors do not have the same shape")
        self.residuals = residuals

    def __init__(self,
                core_shape: Union[int, List[int]],
                core_dtype: Optional[jnp.dtype] = None
                ):

        if core_dtype is not None and not jnp.issubdtype(core_dtype, jnp.floating):
            raise ValueError("dtype of probabilities core must be floating")

        self.core_shape = self._regularize_shape(core_shape)
        self.probabilities = jnp.ndarray(core_shape, core_dtype)
        self.residuals = jnp.ndarray(core_shape, core_dtype)
        self.accumulators = {}

    def build(self)-> ACTStates:
        """
        Builds the actual, usable, act state.
        :return: A ACT state for the situation
        """



        # To build the act state, the collected accumulators
        # basically become the defaults and the

        return ACTStates(
            self.residuals,
            self.probabilities,
            self.accumulators,
            self.accumulators
        )

# Work on the controller





