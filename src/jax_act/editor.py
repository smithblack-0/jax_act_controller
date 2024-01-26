"""
Tools allowing for the
manual editing of an existing ACT session
"""


from typing import Optional, Dict, Any, Tuple, Callable

import jax.tree_util
from enum import Enum
from jax import numpy as jnp
from jax.experimental import checkify

from src.jax_act import utils
from src.jax_act.states import ACTStates
from src.jax_act.types import PyTree
from src.jax_act.immutable import Immutable
from src.jax_act.controller import ACT_Controller

class ErrorModes(Enum):
    standard = 'standard'
    silence = 'silence'

class Editor(Immutable):
    f"""
    ---- Purpose ----

    This class's purpose is to allow the manipulation of
    the states underlying an ACT controller. It is also designed to support
    the programmer by catching boneheaded mistakes that will not become apparent
    until much later.

    ---- Errors modes ----

    Checking for errors in tensor CONTENT is not possible at the moment a method is called
    if you want a method to be jit compatible. This means you cannot
    easily, for instance, ensure that epsilon is probabilistic. Or that various
    other sanity checks based on the content can be passed.

    To handle this, we use jax.experimental's checkify library. This has the behavior
    of delaying the raising of the errors until checkify.check is run. To aid with      
    debugging, when initializing the class you can configure build to run in one of three modes

    These are:
        * {ErrorModes.standard.value}: Raises any errors immediately if possible. When jitted in this mode
                                 you must use checkify.checkify to wrap your jit function.
        * {ErrorModes.silence.value}: Ignores any errors. Presumably faster under eager mode. Also 
                                does not require wrapping with checkify to run.
                
    ---- Leaves and PyTrees ----

    Accumulators may be as simple as just a single tensor of a fixed shape,
    or as complex as PyTrees. See jax.tree_utils for more information.

    Regardless, error messaging and typing will sometimes refer to pytrees.
    If you are not doing anything complex, you can just replace 'pytree' with
    'tensor' and get much the same meaning.

    If you ARE attempting to accumulate pytrees, you will be told what
    PyTree is the problem, but will need to hunt down the leaf in it
    manually.

    ---- How to use: Controller Editing ----

    If you have an existing act process, and you want to edit it, you
    are allowed to do so. The class supports this.

    You can then use one of the setters to set to an existing property. Note that
    you cannot redefine a pytree, so anything set to one must have the same shape, dtype,
    and such.

    Once you are ready, call .build for your output.

    ---- properties ----

    epsilon: The epsilon used to compute the halt threshold
    probabilities: The existing probabilities tensor
    residuals: The existing residuals tensor
    iterations: The existing iterations statistics tensor
    defaults: The existing defaults tensors
    accumulators: The existing accumulators and their values.
    updates: The existing updates and their values.

    ---- set methods ----

    These methods will allow you to manually set the contents of key tensors

    Editing the tensors with these generally requires a really strong understanding
    of how the class runs under the hood.

    However, some safeguards are present. You cannot replace a tensor or tensor collection
    with a tensor or tensor collection of a different shape or dtype

    set_probabilities: Manually replace the probabilities
    set_iterations: Manually redefine the number of iterations statistics container
    set_residuals: Manually redefine the residuals to be particular entities
    set_epsilon: Manually change the epsilon to another floating value.
    set_accumulator: Manually redefine the value of the collection in an accumulator
    set_defaults: Manually redefine what the defaults tensor or tensor collection looks like for
                  an accumulator
    set_updates: Manually set an update accumulator to a valid choice.

    ---- instance options ----

    edit_controller: Edit a controller
    edit_save: Loads a save for editing

    ---- build methods ----

    .build: Build a new controller instance.
    """
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

    @property
    def updates(self)->Dict[str, Optional[PyTree]]:
        return self.state.updates

    # Important validation

    @staticmethod
    def _validate_same_shape(original: jnp.ndarray,
                             new: jnp.ndarray,
                             info: str,
                             ):

        msg = f"""
        The original tensor had shape {original.shape}.
        The proposed new tensor has shape {new.shape}. These did not match.
        """
        msg = utils.format_error_message(msg, info)
        checkify.check(original.shape == new.shape, msg)
    @staticmethod
    def _validate_same_dtype(original: jnp.ndarray,
                             new: jnp.ndarray,
                             info: str,
                             ):
        msg = f"""
        Originally, the tensor had dtype {original.dtype}.
        However, the proposed replacement has dtype {new.dtype}
        """
        msg = utils.format_error_message(msg, info)
        checkify.check(original.dtype == new.dtype, msg)

    @staticmethod
    def _validate_probability(tensor: jnp.ndarray,
                              info: str
                              ):
        # Check low
        msg = """
        A tensor representing probabilities had elements less than zero
        """
        msg = utils.format_error_message(msg, info)
        checkify.check(~jnp.any(tensor < 0), msg)

        # Check high
        msg = """
        A tensor representing probabilities had elements greater than one
        """
        msg = utils.format_error_message(msg, info)
        checkify.check(~jnp.any(tensor > 1.0), msg)
    @staticmethod
    def _validate_is_natural_numbers(tensor: jnp.ndarray,
                                     context: str):
        msg = f"""
        The tensor being processed did not consist of whole
        numbers. That is, numbers >= 0. This is not allowed
        """
        msg = utils.format_error_message(msg, context)
        checkify.check(jnp.all(tensor >= 0), msg)
    def _validate_accumulator_exists(self,
                                     name: str,
                                     context: str)->bool:
        msg = f"""
        Accumulator of name '{name}' was never setup using a define statement
        in the builder. This means you cannot set to it.
        """
        msg = utils.format_error_message(msg, context)
        checkify.check(name in self.defaults, msg)
        return name in self.accumulators
    @staticmethod
    def _validate_same_pytree_structure(
            original: PyTree,
            new: PyTree,
            info: str,
    ):
        """
        Validates that the pytree structure is the same between original
        and new. Raises if not, and uses name parameters to make the
        messages informative.

        :param original: The original pytree feature
        :param new: The new pytree feature
        :param info: The error context to include
        """
        test = utils.are_pytree_structure_equal(original, new)
        msg = f"""
           The original tensor collection or tensor does not have the 
           same tree structure as the new tensor collection or tensor.
           """
        msg = utils.format_error_message(msg, info)
        checkify.check(test, msg)

    @staticmethod
    def _validate_pytree_leaves(
            original: PyTree,
            new: PyTree,
            info: str,
    ):
        """
        Validates that pytree leaves are compatible. The function first
        checks that the original and new pytree leaves are the same types.

        If the types are of tensor type, it also checks that the shape and
        dtypes are the same.

        :param original: The original pytree to test
        :param new: The new pytree to test.
        """

        original_leaves = jax.tree_util.tree_leaves(original)
        new_leaves = jax.tree_util.tree_leaves(new)

        for i, (original_leaf, new_leaf) in enumerate(zip(original_leaves, new_leaves)):
            # Check if leaf has same shape
            msg = f"""
            An issue occurred in leaf {i}
            
            The original shape was {original_leaf.shape}.
            However, the update has shape {new_leaf.shape}
            
            These do not match.
            """
            msg = utils.format_error_message(msg, info)
            checkify.check(original_leaf.shape == new_leaf.shape, msg)

            # Check if leaf has same dtype

            msg = f"""
            An issue occurred in leaf {i}
            
            The original dtype was {original_leaf.dtype}.
            However, the new leaf had dtype {new_leaf.dtype}
            
            These differences are not allowed.
            """
            msg = utils.format_error_message(msg, info)
            checkify.check(original_leaf.dtype == new_leaf.dtype, msg)
    def _execute_validation(self,
                            function: Callable
                            ):
        """
        Executes and handles the various validation modes.

        Validation is packaged into a function, and can then be
        packed into this function. The return will contain
        the new errors list
        :param function: A function that, when called, executes checkify
                         validation.
        :return: An accumulator containing observed errors
        :raise JaxRuntimeError: If in debug mode, and something goes wrong
        """
        if self.error_mode == ErrorModes.standard.value:
            function = checkify.checkify(function)
            errors, _ = function()
            errors.throw()
        else:
            pass
    def set_probabilities(self, values: jnp.ndarray)->'Editor':
        """
        Sets the values of probabilities to something new
        :param values: Values to set
        :return: A new ControllerBuilder with updates applied
        :raises: ValueError, if original and new probability shapes do not match
        :raises: TypeError, if new and original probability types do not match.
        """

        def check_for_errors():
            context_message = "An issue occurred while attempting to set probabilities with an Editor: "
            self._validate_same_shape(self.probabilities, values, context_message)
            self._validate_same_dtype(self.probabilities, values, context_message)
            self._validate_probability(values, context_message)
        self._execute_validation(check_for_errors)
        state = self.state.replace(probabilities=values)
        return Editor(state, error_mode=self.error_mode)

    def set_residuals(self, values: jnp.ndarray)->'Editor':
        """
        Sets the residuals to be equal to particular
        values, and returns a new builder.

        :param values: The values to set the residuals to
        :return: A new ControllerBuilder, with the residuals set
        :raises: ValueError, if the new and original residual shape are not the same
        :raises: TypeError, if the new and original residual do not have the same dtye.
        """
        def check_for_errors():
            context_message = "An issue occurred while attempting to set residuals with an Editor: "
            self._validate_same_shape(self.residuals, values, context_message)
            self._validate_same_dtype(self.residuals, values, context_message)
            self._validate_probability(values, context_message)
        self._execute_validation(check_for_errors)
        state = self.state.replace(residuals=values)
        return Editor(state, self.error_mode)

    def set_iterations(self, values: jnp.ndarray)->'Editor':
        """
        Sets the iteration channel to something new.
        :param values: The iterations tensor to set it to
        :return: The new controller builder
        :raises: If the new and original shape differ
        :raises: If the dtype is not int32
        :raises: If not provided a tensor of natural numbers - numbers >= 0.
        """
        def check_for_errors():
            context_message = "An error occurred while attempting to set the iterations tensor using an Editor"
            self._validate_same_shape(self.iterations, values, context_message)
            self._validate_same_dtype(self.iterations, values, context_message)
            self._validate_is_natural_numbers(values, context_message)
        self._execute_validation(check_for_errors)
        state = self.state.replace(iterations=values)
        return Editor(state, self.error_mode)


    def set_epsilon(self, epsilon: float)->"Editor":
        """
        Sets the epsilon to be a new value.

        :param epsilon: The epsilon to set
        :return: The new controller builder
        :raises: ValueError, if epsilon is not a float
        :raises: ValueError, if epsilon was not between 0-1.
        """

        def check_for_errors():
            context_message = "An error occurred while setting to an epsilon using an Editor"

            # Check epsilon too low
            msg = f"Epsilon was too low. Probabilities should be greater than or equal to zero, got {epsilon}"
            msg = utils.format_error_message(msg, context_message)
            checkify.check(epsilon >= 0.0, msg)

            # Check epsilon too high
            msg = f"Epsilon was too high. Probabilities should be greater than or equal to zero, got {epsilon}"
            msg = utils.format_error_message(msg, context_message)
            checkify.check(epsilon <= 1.0, msg)

        self._execute_validation(check_for_errors)
        state = self.state.replace(epsilon = epsilon)
        return Editor(state, self.error_mode)

    def set_accumulator(self, name: str, value: PyTree)->'Editor':
        """
        Sets an accumulator to be filled with a particular series of values.

        This can only replace like accumulator data with PyTree of the same shape.
        This means those pytrees must be the same tree shape, the leaves must have the same type,
        and tensor leaves must be of the same shape and dtype.

        :param name: The name of the accumulator to set
        :param value: The value to set.
        :return: The new ControllerBuilder
        """

        def check_for_errors():
            context_error_message = "An error occurred while setting to an accumulator with an Editor:"
            if self._validate_accumulator_exists(name, context_error_message):
                self._validate_same_pytree_structure(self.accumulators[name], value, context_error_message)
                self._validate_pytree_leaves(self.accumulators[name], value, context_error_message)
        self._execute_validation(check_for_errors)
        accumulators = self.accumulators.copy()
        accumulators[name] = value
        state = self.state.replace(accumulators=accumulators)
        return Editor(state, self.error_mode)
    def set_defaults(self,
                     name: str,
                     value: PyTree
                     )->'Editor':
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
        """
        def check_for_errors():
            context_error_message = f"An error occurred while setting to a defaults  named '{name}' with an Editor:"
            if self._validate_accumulator_exists(name, context_error_message):
                self._validate_same_pytree_structure(self.accumulators[name], value, context_error_message)
                self._validate_pytree_leaves(self.accumulators[name], value, context_error_message)
        self._execute_validation(check_for_errors)
        defaults = self.defaults.copy()
        defaults[name] = value
        state = self.state.replace(defaults=defaults)
        return Editor(state, self.error_mode)

    def set_updates(self,
                    name: str,
                    values: Optional[PyTree]
                    )->'Editor':
        """
        Sets the update values to new values. This does not
        change any tensor shapes or datatypes.

        This can only replace tensor collections ('pytrees') with other tensors of the same
        shape as the accumulator, or set the updates channel equal to None.

        :param name: The name of the default to replace
        :param values: The pytree to replace it with. May also be None
        :return: A new ControllerBuilder where these replacements had occurred
        """
        def check_for_errors():
            context_error_message = f"An error occurred while setting to an update entry named '{name}' with an Editor"
            self._validate_accumulator_exists(name, context_error_message)

            # Updates can set a none. If not none, validate
            if values is not None:
                self._validate_same_pytree_structure(self.defaults[name], values, context_error_message)
                self._validate_pytree_leaves(self.defaults[name], values, context_error_message)
        self._execute_validation(check_for_errors)
        updates = self.state.updates.copy()
        updates[name] = values
        state = self.state.replace(updates=updates)
        return Editor(state, self.error_mode)

    @classmethod
    def edit_controller(cls,
                        controller: ACT_Controller,
                        error_mode: str = ErrorModes.standard.value)-> 'Editor':
        """
        Opens up a new Editor to edit an existing controller. Returns the
        Editor.

        :param controller: The controller to edit
        :return: The Editor
        """
        modes = [item.value for item in ErrorModes]
        if error_mode not in modes:
            raise ValueError(f"Error mode must have been within {modes}")

        return cls(controller.state, error_mode)

    @classmethod
    def edit_save(cls,
                  save: ACTStates,
                  error_mode: str = ErrorModes.standard.value)-> 'Editor':
        """
        Opens up a builder to edit an existing save from any
        class.

        :param save: The save to edit
        :return: A new ControllerBuilder instance
        """
        modes = [item.value for item in ErrorModes]
        if error_mode not in modes:
            raise ValueError(f"Error mode must have been within {modes}")

        return cls(save, error_mode)

    def save(self)->ACTStates:
        return self.state
    def build(self)->ACT_Controller:
        """
        Build the act controller.
        :return: An ACT Controller instance.
        """
        return ACT_Controller.load(self.state)
    def __init__(self,
                 state: ACTStates,
                 error_mode: str,
                 ):
        """
        This should not be used directly to initialize
        an Editor. Instead, use one of the 'edit' methods
        :param state:
        :param error_mode:
        """
        super().__init__()

        self.error_mode = error_mode
        self.state = state
        self.make_immutable()

def flatten_editor(editor: Editor)->Tuple[Any, Any]:
    state = editor.state
    error_mode = editor.error_mode
    flat_state, tree_def = jax.tree_util.tree_flatten(state)
    return (flat_state, (error_mode, tree_def))

def unflatten_editor(aux_data: Any, flat_state: Any)->Editor:
    error_mode, tree_def = aux_data
    state = jax.tree_util.tree_unflatten(tree_def, flat_state)
    return Editor(state, error_mode)

jax.tree_util.register_pytree_node(Editor, flatten_editor, unflatten_editor)
