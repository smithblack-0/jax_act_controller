"""
EXPERIMENTAL: This module is experimental. No guarentee is yet made that
new versions will not break your existing code.

The ACT layer mixin definition

The mixin and nearby error handling code
is designed to support a programmer attempting to integrate
their project into a specific layer-oriented framework.

It is suppose to be fairly framework-agnostic: So long as the
framework is built on top of jax, it should hopefully work.
"""

from typing import Tuple, Callable, Any

import jax.lax
from src.jax_act.core.controller import ACT_Controller
from src.jax_act.core.types import PyTree
from dataclasses import dataclass, asdict

from .core import AbstractControllerContract, ContractValidationWrapper
from jax import numpy as jnp

# Define and register our internal metadata class.
@dataclass
class ACTMetaData:
    """
    A place for storing metadata
    during iteration.
    """
    def increase_counter(self)->'ACTMetaData':
        return ACTMetaData(self.num_iterations + 1,
                           self.max_num_iterations)
    num_iterations: int
    max_num_iterations: int

def state_flatten(state: ACTMetaData) -> Tuple[Any, Any]:
    state = asdict(state)

    aux_out = []
    values = []
    for key, value in state.items():
        value, aux = jax.tree_util.tree_flatten(value)

        aux_out.append(aux)
        values.append(value)
    return zip(state.keys(), values), aux_out

def state_unflatten(aux_data: Any, flat_state: Any) -> ACTMetaData:

    values = []
    for value, tree_def in zip(flat_state, aux_data):
        value = jax.tree_util.tree_unflatten(tree_def, value)
        values.append(value)
    return ACTMetaData(*values)
jax.tree_util.register_pytree_with_keys(ACTMetaData, state_flatten, state_unflatten)
class AbstractACTTemplate(AbstractControllerContract):
    # TODO:
    #   - update comments to talk about max_iterations
    #   - Add probability depression feature
    """
    The ACT layer based definition. It contains an
    object-oriented, layer-based version of the act process, and
    will assist the user by providing act support methods when
    mixed into an existing framework.

    ----- methods ----

    Contract:

    * make_controller: ABSTRACT. returns a valid controller instance
    * run_iteration: ABSTRACT. Runs a single act layer, and caches then commits the accumulated features.
    * setup_state: ABSTRACT. Can be used to handle any state details.
    * execute_act: CONCRETE. Call this with appropriate parameters to use the layer.

    Utility:

    * new_builder: CONCRETE: Call this with appropriate parameters to go get a builder working.

    ----- Contract -----

    This abstract layer will form the following contract with you, the
    programmer. So long as you implement make_controller such that it returns a
    valid controller, and then implement run_iteration such that it will accept
    and return controller and state, this class will promise to handle loops
    and other details when one of its primary utility method is run.

    ---- Responsibility details: make_controller ----

    The purpose of make_controller is to make a controller with
    accumulators and batch shape setup for accumulation purposes.

    The returned controller:
        * Must be an ACT_Controller instance
        * Must have accumulators to gather

    Failure to conform to this will cause an error to
    be raised in later methods.

    See the method for a full list of contract requirements.

    ---- Responsibilities details: run_iteration ----

    The purpose of run_iteration is to accept the current state and
    run an act update. Minimal restrictions are placed on the coder
    to allow full flexibility.

    This function is what implements the ACTUAL act process,
    sans the while loop. It should consume the current controller
    and current state, and return the new controller and new state.

    Note that importantly, since the controller is immutable, the new
    controller must be a new instance of ACT_Controller

    See the method for a full list of contract requirements.

    ---- Immutable logic ----

    Keep in mind while coding that the classes involved in this
    mechanism are immutable. That means you should never
    write something assuming mutability of the builder or controller
    such as:

    controller.cache_update(...your update)

    Instead always explictly reassign the controller or
    builder to a variable, like follows:

    controller = controller.cache_update(...your_update)

    ---- Code Examples: Simple ACT ----

    This is a simple example of a layer under
    an arbitrary framework which is satisfying
    the contract.

    '''

    from jax_act import ControllerBuilder, ACT_Controller
    from jax import numpy as jnp

    class SimpleActLayer(AbstractLayerMixin, ...your_framework):
        ....
        def update_state(state: jnp.ndarray, input: jnp.ndarray)->jnp.ndarray:
            ... your update function

        def make_output(state: jnp.ndarray)->jnp.ndarray:
            ... your output function

        def make_halting_probabilities(state: jnp.ndarray)->jnp.ndarray:
            ... your halting probability functions

        def make_controller(self,
                            initial_state: jnp.ndarray,
                            num_heads: int,
                            embeddings_width: int,
                            )->ACT_Controller:

            batch_dimension = initial_state.shape[0]
            builder = self.new_builder(batch_dimension)
            builder = builder.define_accumulator_by_shape("state", [batch_dimension, num_heads, embedding_width])
            builder = builder.define_accumulator_by_shape("output", [batch_dimension, num_heads, embedding_width])
            return builder.build()

        def run_layer(self,
                      controller: ACT_Controller,
                      state: jnp.ndarray,
                      input: jnp.ndarray
                      )->Tuple[ACT_Controller, jnp.ndarray]:

            state = self.update_state(state, input)
            output = self.make_output(state)
            halting_probabilities = self.make_halting_probabilities(state)

            controller = controller.cache_update("state", state)
            controller = controller.cache_update("output", output)
            controller = controller.iterate_act(halting_probabilities)
            return controller, state

        # We suppose here we are using something like flax, where
        # you implement your logic using __call__
        def __call__(self, state: jnp.ndarray, input: jnp.ndarray):
            # Runs act
            controller, _ = self.execute_act(tensor, input)

            # Returns result
            state = controller.accumulators["state"]
            output = controller.accumulators["output"]
            return state, output.


    ```
    """
    @staticmethod
    def _is_act_not_complete(combined_state: Tuple[ACT_Controller, PyTree, ACTMetaData]) -> bool:
        # I hate how fudly boolean logic is when using jax
        #
        # WHY must I always use function based composition danget.
        controller, _, metadata = combined_state
        condition = ~controller.is_completely_halted
        condition = jnp.logical_and(condition, metadata.num_iterations < metadata.max_num_iterations)
        return condition

    @staticmethod
    def _while_loop_adapter_factory(layer: 'ContractValidationWrapper',
                                    ) -> Callable[[Tuple[ACT_Controller, PyTree, ACTMetaData]],
                                                  Tuple[ACT_Controller, PyTree, ACTMetaData]]:
        """
        This is primarily an interface between the
        jax.lax.while_loop restrictions and how
        items are passed around to users.
        """
        def run_layer_adapter(state: Tuple[ACT_Controller, PyTree, ACTMetaData]
                              ) -> Tuple[ACT_Controller, PyTree, ACTMetaData]:
            controller, state, metadata = state
            controller, state = layer.run_iteration(controller,
                                                    state)
            return controller, state, metadata.increase_counter()
        return run_layer_adapter


    def execute_act(self,
                    initial_state: PyTree,
                    depression_constant: float = 1.0,
                    max_iterations: int = 10,
                    check_for_errors: bool = True,
                    *args,
                    **kwargs,
                    ) -> Tuple[ACT_Controller, PyTree]:
        """

        Runs a compilable act process so long as the contract
        has been fulfilled between the parent and subclass.

        Hunts down and throws a variety of errors if it has
        not been.

        ---- Responsibilities and contract ----

        This function promises to execute a 'jittable' formulation of adaptive
        computation time (act) when given a layer that follows the ACT Layer Protocol,
        so long as all user components were also jittable.

        Conceptually, the responsibility of the function is to provide a loop
        that can be 'jitted', and to provide reasonable sanity checking regarding
        adherence to the contract. The layer implementing run_iteration and
        make_controller must, in turn, run act itself.

        The layer conforming to the ACT Layer Protocol must in turn perform the
        actual act update process.

        This function behaves like the following python code when
        no errors are encountered.
        ```
        def execute_act(self, initial_state, *args, **kwargs):
            controller = self.make_controller(initial_state, *args, **kwargs)
            state = initial_state
            while not controller.is_halted:
                controller, state = self.run_iteration(controller, state)
            return controller, state
        ```
        :param initial_state: The initial state. This can be just a simple tensor,
                              but can also be a complex pytree if desired.
        :param depression_constant: Multiplies the halting probabilities before they
                                    are incorporated. By scheduling this, we can encourage
                                    a model to check out longer sequences when starting
                                    training
        :param max_iterations: The absolute maximum number of iterations
        :param check_for_errors: Whether to check for errors or charge ahead blindly.
                                 Disabling check_for_errors should remove all errors
                                 overhead when compiled.
        :param *args: Any args to pass into make_controller
        :param **kwargs: Any keyword args to pass into make_controller.
        :return: A controller with the results stored in it, and the final state.
        """
        act_layer = ContractValidationWrapper(self, check_for_errors)
        controller = act_layer.make_controller(initial_state, depression_constant, *args, **kwargs)
        metadata = ACTMetaData(num_iterations=0,
                               max_num_iterations=max_iterations)

        self.setup_lazy_parameters(controller, initial_state)

        wrapped_state = (controller, initial_state, metadata)
        run_layer = self._while_loop_adapter_factory(act_layer)
        controller, final_state, _ = jax.lax.while_loop(self._is_act_not_complete,
                                                      run_layer,
                                                      wrapped_state)
        return controller, final_state

