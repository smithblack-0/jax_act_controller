"""
Creation contains the builders and other mechanisms
needed to create an ACT instance. It also includes
some degree of error checking.

--- PyTrees ----

"""

from typing import Optional, List, Union, Dict, Any, Tuple

import jax.tree_util
import textwrap
import numpy as np
import warnings
from jax import numpy as jnp

from src.jax_act import utils
from src.jax_act.states import ACTStates
from src.jax_act.types import PyTree, PyTreeShapes
from src.jax_act.immutable import Immutable
from src.jax_act.controller import ACT_Controller


class ControllerBuilder(Immutable):
    """
    A Builder class designed to setup a ACT controller, or
    edit an existing one. It has a heavy dose of sanity checking to prevent
    boneheaded mistakes which may not be immediately obvious.

    This is, as most features operating under jax must be, an immutable class.
    When you update your builder, a new builder will be returned to you. You,
    the programmer, must store that builder yourself. This is to ensure compatibility
    with jax.

    ---- Purpose ----

    This class's purpose is to allow the definition and manipulation of
    the states underlying an ACT controller. It is also designed to also support
    the programmer by catching boneheaded mistakes that will not become apparent
    until much later

    ---- Warnings ----

    This class is immutable. Any time you make a change to the builder, a new
    builder will be returned. You will need to assign it to a variable yourself!

    Also, you should never directly call the constructor unless you really know
    what you are doing. Instead, either new_builder or edit_controller should
    be utilized.

    ---- Leaves and PyTrees ----

    Accumulators may be as simple as just a single tensor of a fixed shape,
    or as complex as PyTrees. See jax.tree_utils for more information.

    Regardless, error messaging and typing will sometimes refer to pytrees.
    If you are not doing anything complex, you can just replace 'pytree' with
    'tensor' and get much the same meaning.

    If you ARE attempting to accumulate pytrees, you will be told what
    PyTree is the problem, but will need to hunt down the leaf in it
    manually.

    ---- How to Use: Controller Creation ----

    If you want to create a controller in the first place, you will be wanting
    to use the new_builder method. Define the shape of the batch, or shape
    of the tensors, and the dtype of the probabilities. Your controller is
    returned, and is explicitly bound to the provided batch shape.

    Now, you need to setup the features you wish to accumulate. Usually, they
    are simply tensors, but if you want you can accumulate PyTrees. See jax
    pytrees for definition of that.

    Use one of the various 'define_...' methods to define the name, shapes,
    and default values for your accumulators.

    Once you are ready, call .build() to create a controller. Off you go!

    Generally, you can use this in one of a few ways.

    First, and most importantly, do NOT use the constructor directly. You
    should always use either the new_builder method or the edit_controller
    You can use the builder to setup a act controller in the first
    place. You start


    ---- properties ----

    epsilon: The epsilon used to compute the halt threshold
    probabilities: The existing probabilities tensor
    residuals: The existing residuals tensor
    iterations: The existing iterations statistics tensor
    defaults: The existing defaults tensors
    accumulators: The existing accumulators and their values.
    updates: The existing updates and their values.

    ---- instance methods ----

    These methods will get you an instance so you can start building
    or editing.

    new_builder: Creates your first builder instance to work with
    redefine_controller: Creates a builder from the existing controller state
    redefine_save: Creates a builder from any .save state produced by a class
                from act objects.

    ---- configuration methods ---

    These methods will produce new instances in which the accumulators
    that are being gathered have been fundamentally changed in some way.

    define_accumulator_directly: Use to directly define the initial values of an act accumulator
    define_accumulator_by_shape: Use to define an act accumulator by shape and dtype.
    delete_definition: Use to remove an accumulator from the controller.

    ---- manual methods ----

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


    ---- Example: Configure for a vanilla ACT instance ---

    Scenario: You want to setup an act accumulator, as in the paper.
              You need to define a 'state' and an 'output' accumulator.
              You will have batches

    ```

    batch_shape = 30
    state_shape = 20
    output_shape = 10

    ...

    builder = ControllerBuilder.new_builder(batch_shape)
    builder.define_accumulator_by_shape("state", [batch_shape, state_shape])
    builder.define_accumulator_by_shape("output", [batch_shape, output_shape])

    controller = builder.build()
    ```

    ---- Example: Configure to capture state data produced by some compatible layer

    batch_shape = 10

    process_layer, initial_state = make_recurrent_layer(batch_shape)
    ....

    builder = ControllerBuilder(batch_shape)
    builder.define_accumulator_directly("internal_state", initial_state)

    controller = builder.build()

    ```
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

    # Define class validation

    @staticmethod
    def format_error_message(message: str, context: str)->str:
        message = textwrap.dedent(message)
        message = textwrap.indent(message, "    ")
        return context + "\n" + message

    # Main definition methods.
    #
    # We can actually define the features we wish to accumulate using
    # methods beginning with define.

    def _validate_definition_pytree(self,
                                    pytree: Any,
                                    context: str
                                    ):
        """
        Validates whether or not a proposed definition for an accumulator
        is valid. This includes checking if the structure is actually a
        pytree jax can handle, is the type floating, and does it posses
        correct batch shapes for all leaves.

        :param pytree: The canidate to validate
        :param context: The context to display in the error message.

        :raises: TypeError, if leaves in the pytree are not of tensor type
        :raises: TypeError, if leaves in the pytree are not of floating dtype
        :raises: ValueError, if leaves in the pytree do not batch the batch shape.
        :raises: ValueError, if leaf in the pytree does not have enough dimensions to handle batch dims.
        """



        def validate_tensor_type(leaf: Any):
            if not isinstance(leaf, (jnp.ndarray, np.ndarray)):
                msg = f"""
                Attempt to define an accumulator failed. 
                
                The type of one of the leaves of the provided 
                pytree was not a known tensor
                
                only jax and numpy tensors are supported
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)
        def validate_tensor_floating(leaf: jnp.ndarray):
            if not jnp.issubdtype(leaf, jnp.floating):
                msg = """
                Attempt to define an accumulator failed
                
                The type of the provided tensor was not floating.
                
                This will mean that multiplying by the halting probabilities
                and adding may not work. As a result, it is disallowed. 
                
                Fix this by converting to a floating dtype first.
                """
                msg = textwrap.dedent(msg)
                raise TypeError(msg)
        def validate_tensor_batch_shape(leaf: jnp.ndarray):
            batch_shape = self.probabilities.shape
            batch_dim_length = len(batch_shape)

            if batch_dim_length > len(leaf.shape):
                msg = f"""
                Attempt to define an accumulator failed.
                
                At least one of the leaves had a tensor with 
                number of dimensions less than the batch shape. The
                failing tensor had shape '{leaf.shape}'
                
                Make sure to place batch dimensions at the beginning
                of your tensors
                """
                msg = textwrap.dedent(msg)
                raise ValueError(msg)

            tensor_shape = leaf.shape[:batch_dim_length]
            if tensor_shape != batch_shape:
                msg = f"""
                Attempt to define an accumulator failed
                
                One of the leaves during traversal had 
                batch dimensions '{tensor_shape}'.
                
                These do not match the defined batch
                shape of '{batch_shape}'
                """
                msg = textwrap.dedent(msg)
                raise ValueError(msg)
        def validate_leaf(leaf):


            validate_tensor_type(leaf)
            validate_tensor_floating(leaf)
            validate_tensor_batch_shape(leaf)

        jax.tree_util.tree_map(validate_leaf, pytree)


    def _define(self,
                name: str,
                definition: PyTree,
                )->'ControllerBuilder':
        """
        A helper method. Attempts to define an
        accumulator with a shape that is different, or
        a new accumulator.

        :param name: The name to set
        :param definition: What to set it to
        :return: A new ControllerBuilder
        """

        self._validate_definition_pytree(definition)

        defaults = self.defaults.copy()
        accumulators = self.accumulators.copy()
        updates = self.updates.copy()

        defaults[name] = definition
        accumulators[name] = definition
        updates[name] = None

        state = self.state.replace(defaults=defaults,
                                   accumulators=accumulators,
                                   updates=updates)
        return ControllerBuilder(state)

    def _delete_definition(self, name: str)->'ControllerBuilder':
        """
        Returns a new controller builder with the definition deleted
        :param name: The name to delete

        :return: The new ControllerBuilder
        """

        defaults = self.defaults.copy()
        accumulators = self.accumulators.copy()
        updates = self.updates.copy()

        defaults.pop(name)
        accumulators.pop(name)
        updates.pop(name)

        state = self.state.replace(defaults=defaults,
                                   accumulators=accumulators,
                                   updates=updates)

        return ControllerBuilder(state)

    def define_accumulator_directly(self,
                                   name: str,
                                   definition: PyTree,
                                   )->'ControllerBuilder':
        """
        Directly define the default state of an accumulator,
        including it's shape. The passed tensor, tensors, or
        PyTree become the defaults and accumulator values.

        If the accumulator is already setup, it is overwritten.
        This will NOT, however, overwrite other quantities such
        as probabilities and residuals.

        Example:

        ```
        from jax import numpy as jnp
        ...

        attn_initial_state  = {}
        attn_initial_state["matrix"] = jnp.zeros([batch_shape, head_dim, embeddings, embeddings])
        attn_initial_state["normalizer] = jnp.zeros([batch_shape, head_dim, embeddings]

        builder = builder.define_accumulator_directly("attn_state", attn_initial_state)
        ```

        :param name: The name of the accumulator to setup
        :param definition: What collection of tensors to start the accumulator out with.
        :return: A new ControllerBuilder.
        :raises: TypeError, if the definition is flawed.
        :raises: ValueError, if the definition is flawed.
        """

        if name in self.defaults:
            warnings.warn("Accumulator and defaults are being overwritten")
        try:
            new_builder = self._define(name, definition)
        except Exception as err:
            msg = f""" 
            An issue occurred while trying to define an accumulator
            named '{name}'
            """
            msg = textwrap.dedent(msg)
            raise err.__class__(msg) from err
        return new_builder

    def define_accumulator_by_shape(self,
                                    name: str,
                                    shapes: Union[PyTreeShapes, List[int]],
                                    dtype: jnp.dtype = jnp.float32,
                                    fill: float = 0.0,
                                    )->'ControllerBuilder':
        """
        Define a new accumulator by providing the
        shape the accumulator should be, the dtype, and
        the fill.

        The function will track down the list of ints provided in shapes,
        and assume that they should be used to construct a new pytree with the
        given shape. Then, it will make tensors of those shapes, in the same
        positions on a pytree, if relevant.

        Examples

        ```
        # Simple case
        ...
        state_shape = [batch_dim, embeddings_dim]
        builder = builder.define_by_shape("state", state_shape)

        # Complex case, with pytrees
        ...
        state_shape = {}
        state_shape["matrix"] = [batch_dim, head_dim, embedding_dim, embedding_dim]
        state_shape["normalizer"] = [batch_dim, head_dim, embedding_dim]

        builder = builder.define_by_shape("state", state_shape)
        ```

        :param name: The name of the accumulator to make
        :param shapes: A PyTree whose leaves are lists of ints, or a list of int
        :param dtype: A dtype to make the accumulators. Defaults to float32
        :param fill: A fill value to make the accumulators. Defaults to 0.0
        :return: A new ControllerBuilder
        :raise: ValueError, if the provided shape did not match the batch shapes.
        :raise: TypeError, if you provide a bad dtype
        """
        if name in self.defaults:
            warnings.warn("Accumulator and defaults are being overwritten")

        def is_leaf(node: Any)->bool:
            # Test to find out if a
            # node is a leaf
            #
            # Our leaves should be lists of ints, but
            # we also need to traverse lists like normal.
            #
            # As a result, when we see a list, we look ahead
            # to find out if it is filled with ints, and
            # return true in those cases.

            contains_integers = False
            if isinstance(node, list):
                for item in node:
                    if not isinstance(item, int):
                        return False
                return True
            return False
        def make_tensor(leaf: List[int])->jnp.ndarray:
             # Convert a list of ints into a tensor
             # of the same shape
             if not is_leaf(leaf):
                 msg = f"""
                 The shape definition collection is corrupt. 
                 
                 A type of {type(leaf)} was reached, but this is not
                 a valid list of integers
                 """
                 msg = textwrap.dedent(msg)
                 raise TypeError(msg)
             return jnp.full(leaf, fill_value=fill, dtype=dtype)

        # BEGIN: Main logic

        definition = jax.tree_util.tree_map(make_tensor, shapes, is_leaf=is_leaf)
        try:
            new_builder = self._define(name, definition)
        except Exception as err:
            msg = f""" 
            An issue occurred while trying to define an accumulator
            named '{name}'
            """
            msg = textwrap.dedent(msg)
            raise ValueError(msg) from err
        return new_builder

    def delete_definition(self, name: str)->'ControllerBuilder':
        """
        Delete an existing accumulator definition.

        Return a new controller with the deletion in place
        :param name: The name of the controller to delete
        :return: A new ControllerBuilder instance
        """
        if name not in self.defaults:
            msg = f"""\
            Accumulator of name '{name}' does not exist, and thus
            cannot be deleted.
            """
            msg = textwrap.dedent(msg)
            raise KeyError(msg)

        return self._delete_definition(name)

    # Instance making methods
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
    def redefine_controller(cls, controller: ACT_Controller)-> 'ControllerBuilder':
        """
        Opens up a new builder to edit an existing controller.

        Returns, as normal, the builder.
        :param controller: The controller to edit
        :return: The builder
        """
        return cls(controller.state)

    @classmethod
    def redefine_save(cls, save: ACTStates)-> 'ControllerBuilder':
        """
        Opens up a builder to edit an existing save from any
        class.

        :param save: The save to edit
        :return: A new ControllerBuilder instance
        """
        return ControllerBuilder(save)

    def build(self)->ACT_Controller:
        """
        Build the act controller
        :return: An ACT Controller instance.
        """
        return ACT_Controller(self.state)
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

def flatten_builder(builder: ControllerBuilder)->Tuple[Any, Any]:
    state = builder.state
    flat_state, tree_def = jax.tree_util.tree_flatten(state)
    return (flat_state, tree_def)

def unflatten_builder(aux_data: Any, flat_state: Any)->ControllerBuilder:
    state = jax.tree_util.tree_unflatten(aux_data, flat_state)
    return ControllerBuilder(state)

jax.tree_util.register_pytree_node(ControllerBuilder, flatten_builder, unflatten_builder)
