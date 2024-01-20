"""
Creation contains the builders and other mechanisms
needed to create an ACT instance. It also includes
some degree of error checking.
"""

from typing import Optional, List, Union, Dict, Any

import jax.tree_util
import textwrap
import numpy as np
import warnings
from jax import numpy as jnp

from src import utils
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

    # Manual build properties and
    # methods.

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
        return self.state.defaults

    @property
    def accumulators(self)->Dict[str, PyTree]:
        return self.state.accumulators

    # Important setters

    @staticmethod
    def _validate_set_shape(original: jnp.ndarray,
                            new: jnp.ndarray,
                            field_name: str):

        if original.shape != new.shape:
            msg = f"""
            Attempt to set feature named '{field_name}' with original shape {original.shape}.
            
            The proposed new values have shape {new.shape}. These did not match.
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)

    @staticmethod
    def _validate_set_dtype(original: jnp.ndarray,
                            new: jnp.ndarray,
                            field_name: str):
        if original.dtype != new.dtype:
            msg = f"""
            Attempt to set feature with wrong dtype. 
            
            Field of name {field_name} originally had dtype {original.dtype}.
            
            However, the proposed update has dtype {new.dtype}
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)


    def set_probabilities(self, values: jnp.ndarray)->'ControllerBuilder':
        """
        Sets the values of probabilities to something new
        :param values: Values to set
        :return: A new ControllerBuilder with updates applied
        :raises: ValueError, if original and new probability shapes do not match
        :raises: TypeError, if new and original probability types do not match.
        """

        self._validate_set_shape(self.probabilities, values, field_name="probabilities")
        self._validate_set_dtype(self.probabilities, values, field_name="probabilities")

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
        self._validate_set_shape(self.residuals, values, field_name="residuals")
        self._validate_set_dtype(self.residuals, values, field_name="residuals")

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

        self._validate_set_shape(self.iterations, values, field_name="iterations")
        self._validate_set_dtype(self.iterations, values, field_name="iterations")

        state = self.state.replace(iterations=values)
        return ControllerBuilder(state)

    def set_epsilon(self, epsilon: float)->"ControllerBuilder":
        """
        Sets the epsilon to be a new value.

        :param epsilon: The epsilon to set
        :return: The new controller builder
        :raises: ValueError, if epsilon is not a float
        :raises: ValueError, if epsilon was not between 0-1.
        """

        if not isinstance(epsilon, float):
            raise ValueError("Epsilon was not a float")
        if (epsilon > 1.0) | (epsilon < 0.0):
            raise ValueError("Epsilon was not between 0 and 1")

        state = self.state.replace(epsilon = epsilon)
        return ControllerBuilder(state)

    # We need some more helper methods to validate
    # pytrees

    @staticmethod
    def _validate_same_pytree_structure(
            original: PyTree,
            new: PyTree,
            field_name: str,
            key_name: str
            ):
        """
        Validates that the pytree structure is the same between original
        and new. Raises if not, and uses name parameters to make the
        messages informative.

        :param original: The original pytree feature
        :param new: The new pytree feature
        :param field_name: The name of the field being set to
        :param item_name: The name of the item in the dictionary
        :raises: ValueError, if the tree structures are different
        """
        if not utils.are_pytree_structure_equal(original, new):
            msg = f"""
            The attempt to set to key '{key_name}' in field '{field_name} failed.
            
            The original and new pytrees had different tree structures and are not 
            compatible. If this is intended, use a define statement instead.
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg)

    @staticmethod
    def _validate_pytree_leaves(
        original: PyTree,
        new: PyTree,
        field_name: str,
        key_name: str,
        ):
        """
        Validates that pytree leaves are compatible. The function first
        checks that the original and new pytree leaves are the same types.

        If the types are of tensor type, it also checks that the shape and
        dtypes are the same.

        :param original: The original pytree to test
        :param new: The new pytree to test.
        :param field_name: The name of the field being checked
        :param key_name: The name of the key in the field we are setting.
        :raises: ValueError, if the leaf types are different
        :raises: ValueError, if the leaf shapes are different
        :raises: ValueError, if the leaf dtypes are different
        """

        def check_function(original_leaf: Any,
                           new_leaf: Any
                           ):
            """
            A check function that inspects two leafs and
            considers whether or not the new leaf
            is a compatible replacement of the original.

            :param original_leaf: The original pytree leaf
            :param new_leaf: The new pytree leaf.
            :raises: ValueError, if the leaf types are different
            :raises: ValueError, if the leaves are tensors with different shapes
            :raises: ValueError, if the leaves are tensors with different dtypes
            """

            if type(original_leaf) != type(new_leaf):
                msg = f"""
                The type of a leaf in pytree named '{key_name}' of field {field_name}
                is different between the original and new tree. This is not allowed
                """
                msg = textwrap.dedent(msg)
                raise ValueError(msg)
            if isinstance(original_leaf, (jnp.ndarray, np.ndarray)):
                if original_leaf.shape != new_leaf.shape:
                    msg = f"""
                    The shaoe of some tensors in item '{key_name}' of field '{field_name}' is 
                    different. The original tensor had shape {original_leaf.shape}, but the 
                    new tensor has shape {new_leaf.shape}.
                    """
                    msg = textwrap.dedent(msg)
                    raise ValueError(msg)
                if original_leaf.dtype != new_leaf.dtype:
                    msg = f"""
                    The dtype of some tensors in item '{key_name}' of field '{field_name}' do not match.
                    
                    The original tensor entry had dtype of {original_leaf.dtype}, but the new update
                    has a dtype of {new_leaf.dtype}
                    """
                    msg = textwrap.dedent(msg)
                    raise ValueError(msg)
        utils.merge_pytrees(check_function, original, new)

    def set_accumulator(self, name: str, value: PyTree)->'ControllerBuilder':
        """
        Sets an accumulator to be filled with a particular series of values.

        This can only replace like accumulator data with PyTree of the same shape.
        This means those pytrees must be the same tree shape, the leaves must have the same type,
        and tensor leaves must be of the same shape and dtype.

        :param name: The name of the accumulator to set
        :param value: The value to set.
        :return: The new ControllerBuilder
        :raises: KeyError, if you are trying to set to an accumulator that was never defined.
        :raises: ValueError, if your pytree is not compatible.
        """
        if name not in self.accumulators:
            msg = f"""
            Accumulator of name '{name}' was never setup using a define statement.
            
            Use one of the 'define' methods to setup the defaults and the accumulators.
            This method only allows replacement of already existing features.
            """
            msg = textwrap.dedent(msg)
            raise KeyError(msg)

        self._validate_same_pytree_structure(self.accumulators,
                                             value,
                                             field_name="accumulators",
                                             key_name=name)
        self._validate_pytree_leaves(self.accumulators,
                                     value,
                                     field_name="accumulators",
                                     key_name=name)

        accumulators = self.accumulators.copy()
        accumulators[name] = value
        state = self.state.replace(accumulators=accumulators)
        return ControllerBuilder(state)

    def set_defaults(self,
                     name: str,
                     value: PyTree
                     )->'ControllerBuilder':
        """
        Sets the default values for accumulators to new values. This does not
        change any tensor shapes or datatypes.

        This can only replace pytree defaults data with other PyTrees of the same shape.
        This means those pytrees must be the same tree shape, the leaves must have the same type,
        and tensor leaves must be of the same shape and dtype.

        Additionally, the default must have been defined using one of the 'define' methods.

        :param name: The name of the default to replace
        :param value: The pytree to replace it with
        :return: A new ControllerBuilder where these replacements had occurred
        :raises: KeyError, if you are trying to set a default that was never defined
        :raises: ValueError, if the pytrees are not compatible.
        """

        if name not in self.defaults:
            msg = f"""
            Defaults of name '{name}' was never setup using a define statement.
            
            Use one of the 'define' methods to setup the defaults and the accumulators.
            This method only allows replacement of already existing features.
            """
            msg = textwrap.dedent(msg)
            raise KeyError(msg)

        self._validate_same_pytree_structure(self.defaults,
                                             value,
                                             field_name="defaults",
                                             key_name=name)
        self._validate_pytree_leaves(self.defaults,
                                     value,
                                     field_name="defaults",
                                     key_name=name
                                     )

        defaults = self.defaults.copy()
        defaults[name] = value
        state = self.state.replace(defaults=defaults)
        return ControllerBuilder(state)




    # Main build methods and objects. These are used to
    # actually create accumulators in the first place.

    def define_accumulator(self,
                           name: str,
                           definition: PyTree,
                           )->'ControllerBuilder':
        """
        Directly define the default state of an accumulator,
        including it's shape. The passed tensors, or pytree,
        becomes the default.

        If the accumulator is already setup, it is overwritten

        :param name: The name of the accumulator to setup
        :param definition: What to start the accumulator with
        :return: A new ControllerBuilder.
        """

        if name in self.defaults:
            warnings.warn("Accumulator and defaults are being overwritten")
        defaults = self.defaults.copy()
        accumulato

    def define_accumulator_by_shape(self,
                                    name: str,
                                    definition: PyTree,
                                    dtype: Optional[jnp.dtype] = None
                                   ):



    # Creation, initialization, and editing

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





