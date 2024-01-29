"""
The ACT layer protocol and usage definition.

This defines the protocol for making a layer that
can easily be incorporated into an act process regardless
of framework, and which can still be compiled.

---- contents ----

Objects of interest are

ACTLayerProtocol: Defines the protocol and restrictions for an act compatible layer, whatever framework is being used
execute_act: When passed a layer satisfying the protocol, an initial state, and whatever construction arguments
             you want it will run an act iteration in a compile-friendly manner.
"""

from typing import Tuple, Union, List, Optional, Callable, Dict, Protocol, Any

import jax.lax
import textwrap

from src.jax_act import utils
from src.jax_act.controller import ACT_Controller
from src.jax_act.types import PyTree
from jax import numpy as jnp


# Define the layer protocol
class ACTLayerProtocol(Protocol):
    """
    The ACT layer protocol definition. Defines what is
    needed from a layer in order to be usable in an
    act function.

    ----- methods ----

    Two methods must be implemented by a class
    satisfying this protocol. These are:

    * make_controller: returns a valid controller instance
    * run_layer: Runs a single act layer, and caches then commits the accumulated features.

    ---- caution about immutability -----

    Be careful you are using the immutable ControllerBuilder and Controller
    objects right. You should never write any code that assumes mutability
    such as:

    controller.cache_update(... your update)

    Instead, do:

    controller = controller.cache_update(... your update)

    Error messages exist suggesting when you are making this mistake.

    ---- details: make_controller ----

    Make controller must return an ACT_Controller instance, and the
    controller must not be empty. It also should be compatible with the
    downstream run_layer call.


    A reasonable definition to hold two accumulators,
    'output' and 'state', might look like:

    ```
    def make_controller(self,
                        initial_state: jnp.ndarray,
                        num_heads: int,
                        embeddings_width: int,
                        )->ACT_Controller:
        batch_dimension = initial_state.shape[0]
        builder = ControllerBuilder.new_builder(batch_dimension)
        builder = builder.define_accumulator_by_shape("state", [batch_dimension, num_heads, embedding_width])
        builder = builder.define_accumulator_by_shape("output", [batch_dimension, num_heads, embedding_width])
        return builder.build()

    ```

    However, note that initial state need not be a tensor. You may
    very well want your own custom structure. That is fine. So long as
    you provide a valid jax PyTree, and know how to use it to make
    a controller, the function will run just fine.

    ---- details: run_layer -----

    This should, conceptually, run a single iteration of the
    act process. This means accepting the controller and current
    state, updating the current state, making updates and caching
    them in appropriate accumulators, then committing them with
    halting probabilities.

    A reasonable example involving internal helper functions "update_state" and
    "get_halting_probabilities", and accumulating "state" and "output" tensors is
    shown below.

    ```
    def run_layer(self,
                  controller: ACT_Controller,
                  state: jnp.ndarray
                  )->Tuple[ACT_Controller, jnp.ndarray]:

        state = self.update_state(state)
        output = self.make_output(state)
        halting_probabilities = self.make_halting_probabilities(state)

        controller = controller.cache_update("state", state)
        controller = controller.cache_update("output", output)
        controller = controller.iterate_act(halting_probabilities)
        return controller, state
    ```

    """
    def make_controller(self,
                        initial_state: PyTree,
                        *args,
                        **kwargs
                        ) -> ACT_Controller:
        pass

    def run_layer(self,
                  controller: ACT_Controller,
                  state: PyTree,
                  ) -> Tuple[ACT_Controller, PyTree]:
        pass


# The wrapper provides validation and a little bit of
# adapter functionality.
class _ACTWrapper:
    def __init__(self,
                 layer: ACTLayerProtocol,
                 check_errors: bool
                 ):
        self.layer = layer
        self.check_errors = check_errors

    def _execute_validation(self,
                            function: Callable
                            ):
        """
        Executes and handles the various validation modes.

        Validation is packaged into a function, and depending
        on the mode may or may not be executed
        """
        if self.check_errors is True:
            function()

    @staticmethod
    def _validate_is_controller(item: Any, context: str):
        if not isinstance(item, ACT_Controller):
            msg = f"""
            Expected to be given a result of type {type(ACT_Controller)}. 
            However, we instead got a result of type {type(item)}
            """
            msg = utils.format_error_message(msg, context)
            raise TypeError(msg)

    @staticmethod
    def _validate_controller_not_empty(item: ACT_Controller, context: str):
        if len(item.accumulators) == 0:
            msg = f"""
            An empty controller was defined. This cannot hold anything, and so is
            not allowed.
            
            It is possible you did not remember that the builder is not mutable. Did you use the 
            proper syntax of:
            
            builder = builder.define_...your definition
            
            Rather than:
            
            builder.define_...your definition
            
            when defining your accumulators using the builder?
            """
            msg = utils.format_error_message(msg, context)
            raise RuntimeError(msg)

    @staticmethod
    def _validate_is_tuple(item: Any, context: str):
        if not isinstance(item, tuple):
            msg = f"""
            Expected to see tuple, but instead got type {type(item)}
            """
            msg = utils.format_error_message(msg, context)
            raise TypeError(msg)

    @staticmethod
    def _validate_length_two(item: Any, context: str):
        if len(item) != 2:
            msg = f"""
            Expected collection to have length of two. One for
            controller, one for state
            """
            msg = utils.format_error_message(msg, context)
            raise RuntimeError(msg)

    @staticmethod
    def _validate_is_novel_controller(original_controller: ACT_Controller,
                                      new_controller: ACT_Controller,
                                      context: str):
        if new_controller is original_controller:
            msg = f"""
            Original and new controller were the same. This
            is not allowed.
            
            This is likely to occur if you did not replace
            your original controller. Are you using functional 
            notation like the following?
            
            controller = controller.cache_state(...your update)
            controller = controller.iterate_act(...your halting probabilities)
            
            Rather than:
            
            controller.cache_state(...your update)
            controller.iterate_act(...your halting probabilities)
            
            """
            msg = utils.format_error_message(msg, context)
            raise RuntimeError(msg)

    @staticmethod
    def _validate_all_act_updates_committed(controller: ACT_Controller,
                                            context: str):
        if controller.updates_ready_to_commit:
            msg = f"""
            All accumulators were filled, but you never committed the 
            updates. You must commit all updates using:
            
            controller = controller.iterate_act(halting_probabilities) 
            
            rather than:
            
            controller.iterate_act(halting_probabilities)
            """
            msg = utils.format_error_message(msg, context)
            raise RuntimeError(msg)
        if jnp.logical_and(controller.has_cached_updates, ~controller.updates_ready_to_commit):
            msg = f"""
            Controller had uncommitted updates and is not ready to commit. Make sure
            you use controller.cache_update to cache an update for every accumulator you defined.
            Also, make sure you use the immutable syntax:
            
            controller = controller.cache_update(...your update)
            
            Rather than:
            
            controller.cache_update(...your_update)
            """
            msg = utils.format_error_message(msg, context)
            raise RuntimeError(msg)


    def make_controller(self, initial_state: PyTree, *args, **kwargs) -> ACT_Controller:
        controller = self.layer.make_controller(initial_state, self.check_errors, *args, **kwargs)
        def validate():
            context_error_msg = f"An issue occurred while trying to make an act controller using the user layer"
            self._validate_is_controller(controller, context_error_msg)
            self._validate_controller_not_empty(controller, context_error_msg)
        self._execute_validation(validate)
        return controller


    def run_layer(self,
                  controller: ACT_Controller,
                  state: PyTree
                  ) -> Tuple[ACT_Controller, PyTree]:

        update = self.layer.run_layer(controller, state)
        def validate():
            # Perform return formatting validation
            context_message = "An issue occurred while validating the execution of run_layer"
            self._validate_is_tuple(update, context_message)
            self._validate_length_two(update, context_message)

            # Validate the controller
            new_controller, _ = update
            self._validate_is_controller(new_controller, context_message)
            self._validate_is_novel_controller(controller, new_controller, context_message)
            self._validate_all_act_updates_committed(controller, context_message)
        self._execute_validation(validate)
        return update

    def __call__(self,
                 state: Tuple[ACT_Controller, PyTree]
                 ) -> Tuple[ACT_Controller, PyTree]:
        """
        This is primarily an interface between the
        jax.lax.while_loop restrictions and how
        items are passed around to users.
        """
        controller, state = state
        controller, state = self.run_layer(controller, state)
        return controller, state


## DEFINE: Helpers ##

def _is_act_not_complete(combined_state: Tuple[ACT_Controller, PyTree]) -> bool:
    controller, _ = combined_state
    return ~controller.is_completely_halted


def execute_act(layer: ACTLayerProtocol,
                initial_state: PyTree,
                check_for_errors: bool = True,
                *args,
                **kwargs,
                ) -> Tuple[ACT_Controller, PyTree]:
    """
    Runs a compilable act process so long as the provided
    layer follows the ACTLayerProtocol and an initial state
    is provided. Optionally, suppress errors.

    ---- Python equivalency ----

    This code has a python equivalent. In particular, it behaves like:

    ```
    initial_state = ...

    state = initial_state
    controller = layer.make_controller(state)
    while not controller.is_halted:
        controller, state = layer.run_layer(controller, state)
    ```

    :param layer: A layer implementing the ACTLayerProtocol
    :param initial_state: The initial state. Usually just a tensor, but can be a pytree
    :param check_for_errors: Whether to check for errors or charge ahead blindly.
    :return: A controller with the results stored in it, and the final state.
    """
    act_layer = _ACTWrapper(layer, check_for_errors)
    controller = act_layer.make_controller(initial_state, *args, **kwargs)
    wrapped_state = (controller, initial_state)
    controller, final_state = jax.lax.while_loop(_is_act_not_complete,
                                                 act_layer,
                                                 wrapped_state
                                                 )
    return controller, final_state
