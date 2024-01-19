"""
Creation contains the builders and other mechanisms
needed to create an ACT instance. It also includes
some degree of error checking.
"""

from typing import Optional, List, Union, Dict

import jax.tree_util
import textwrap
from jax import numpy as jnp

from src.states import ACTStates
from src.types import PyTree
from src.immutable import Immutable
from src.controller import ACT_Controller


class ControllerBuilder(Immutable):
    """
    A builder class for setting up a act controller.

    This is an immutable builder that lets you setup,
    modify, and otherwise configure your act situation.

    Being immutable, any change returns a new controller
    builder and it is up to you, the programmer, to ensure
    you keep the new one around.
    """

    #Item getters

    @property
    def epsilon(self)->float:
        return self.state.epsilon
    @property
    def probabilities(self)->jnp.ndarray:
        return self.state.probabilities

    @property
    def residuals(self)->jnp.ndarray:
        return self.state.residuals

    @property
    def iterations(self)->jnp.ndarray:
        return self.state.iterations

    @property
    def defaults(self)->Dict[str, PyTree]:
        return self.state.accumulators

    @property
    def accumulators(self)->Dict[str, PyTree]:
        return self.state.accumulators

    # Important setters
    def set_probabilities(self, values: jnp.ndarray)->'ControllerBuilder':
        """
        Sets the values of probabilities to something new
        :param values: Values to set
        :return: A new ControllerBuilder with updates applied
        :raises: ValueError, if original and new probability shapes do not match
        :raises: TypeError, if new and original probability types do not match.
        """
        if self.probabilities.shape != values.shape:
            msg = f"""
            Attempt to set new probabilities of shape '{values.shape}'
            However, existing batch shape is '{self.probabilities.shape}'.
            
            These do not match
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)
        if self.probabilities.dtype != values.dtype:
            msg = f"""
            Attempt to set new probabilities with dtype of {values.dtype},
            which is different from {self.probabilities.dtype}
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)
        state = self.state.replace(residuals=values)
        return ControllerBuilder(state)

    def set_residuals(self, values: jnp.ndarray)->'ControllerBuilder':
        """
        Sets the residuals to be equal to particular
        values, and returns a new builder.

        :param values: The values to set the residuals to
        :return: A new ControllerBuilder, with the residuals set
        :raises: ValueError, if the new and original residual shape are not the same
        :raises: TypeError, if the new and original residual do not have the same dtye.
        """
        if self.residuals.shape != values.shape:
            msg = f"""
            Attempt to set new residuals of shape {values.shape}. 
            
            However, existing batch shape is {self.residuals.shape}
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)
        if self.residuals.dtype != values.dtype:
            msg = f"""
            Attempt to set new residuals of dtype {values.dtype}. 
            
            However, existing dtype was {self.residuals.dtype}
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)
        state = self.state.replace(residuals=values)
        return ControllerBuilder(state)

    def set_iterations(self, values: jnp.ndarray)->'ControllerBuilder':
        """
        Sets the iteration channel to something new.
        :param values: The iterations tensor to set it to
        :return: The new controller builder
        :raises: If the new and original shape differ
        :raises: If the dtype is not int32
        """
        if self.residuals.shape != values.shape:
            msg = f"""
            Attempt to set new iterations tensor of shape {values.shape}. 

            However, existing batch shape is {self.iterations.shape}
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)
        if values.dtype != jnp.int32:
            msg = f"""
            Attempt to set new interations tensor of wrong dtype. 
            
            Got tensor of dtype {values.dtype}, but needed int32.
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)
        state = self.state.replace(iterations=values)
        return ControllerBuilder(state)

    def set_epsilon(self, epsilon: float)->ControllerBuilder:


    # Factories and other assistance.



    # Creation and editing
    @classmethod
    def new_builder(cls,
                    batch_shape: Union[int, List[int]],
                    core_dtype: Optional[jnp.dtype] = None,
                    epsilon: float = 1e-4,
                    )->'ControllerBuilder':
        """
        Creates a new builder you can then
        edit.

        :param batch_shape: The batch shape. Can be an int, or a list of ints
        :param core_dtype: The dtype of data
        :param epsilon: The epsilon for the act threshold
        :return: A StateBuilder instance
        """

        if core_dtype is not None and not jnp.issubdtype(core_dtype, jnp.floating):
            raise ValueError("dtype of probabilities core must be floating")
        if (epsilon >= 1.0) or (epsilon <=0.0):
            raise ValueError("epsilon insane: Not between 0 and 1")

        probabilities = jnp.zeros(batch_shape, core_dtype)
        residuals = jnp.zeros(batch_shape, core_dtype)
        iterations = jnp.zeros(batch_shape, jnp.int32)

        accumulators = {}
        defaults = {}
        updates = {}

        state = ACTStates(
            epsilon=epsilon,
            probabilities=probabilities,
            iterations=iterations,
            residuals=residuals,
            accumulators=accumulators,
            defaults=defaults,
            updates=updates,
        )

        return cls(state)

    @classmethod
    def edit_controller(cls, controller: ACT_Controller)->'ControllerBuilder':
        """
        Opens up a new builder to edit an existing controller.

        Returns, as normal, the builder.
        :param controller: The controller to edit
        :return: The builder
        """
        return cls(controller.state)
    def __init__(self,
                 state: ACTStates
                 ):
        """
        Do not use this class to get your initial act builder.

        Instead, use method 'new_builder' to create a new
        builder and 'edit_controller' to create a builder to
        edit an existing controller.
        """
        super().__init__()
        self.state = state
        self.make_immutable()

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

    def __init__(self,
                 batch_shape: Union[int, List[int]],
                 core_dtype: Optional[jnp.dtype] = None,
                 epsilon: float = 1e-4,
                 ):

        if core_dtype is not None and not jnp.issubdtype(core_dtype, jnp.floating):
            raise ValueError("dtype of probabilities core must be floating")

        probabilities = jnp.zeros(batch_shape, core_dtype)
        residuals = jnp.zeros(batch_shape, core_dtype)
        iterations = jnp.zeros(batch_shape, jnp.int32)

        accumulators = {}
        defaults = {}
        updates = {}

        self.state = ACTStates(
            epsilon=epsilon,
            probabilities=probabilities,
            iterations=iterations,
            residuals=residuals,
            accumulators=accumulators,
            defaults=defaults,
            updates=updates,
        )


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
                core_dtype: Optional[jnp.dtype] = None,
                epsilon: float = 1e-4,
                ):

        if core_dtype is not None and not jnp.issubdtype(core_dtype, jnp.floating):
            raise ValueError("dtype of probabilities core must be floating")

        self.epsilon = epsilon
        self.dtype = core_dtype


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

    def __init__(self, epsilon: float = 1e-4):



# Work on the controller





