"""
The ACT layer mixin definition

The mixin and nearby error handling code
is designed to support a programmer attempting to integrate
their project into a specific layer-oriented framework.

It is suppose to be fairly framework-agnostic: So long as the
framework is built on top of jax, it should hopefully work.
"""

from typing import Tuple, Callable, Protocol, Any

import jax.lax
from abc import ABC, abstractmethod
from src.jax_act import utils
from src.jax_act.controller import ACT_Controller
from src.jax_act.types import PyTree
from jax import numpy as jnp

class AbstractLayerMixin(ABC):
    """
    The ACT layer based definition. It contains an
    object-oriented, layer-based version of the act process, and
    will assist the user by providing act support methods when
    mixed into an existing framework.

    ----- methods ----

    * make_controller: ABSTRACT. returns a valid controller instance
    * run_iteration: ABSTRACT. Runs a single act layer, and caches then commits the accumulated features.
    * get_builder: CONCRETE: Call this with appropriate parameters to go get a builder working.

    * run_act: CONCRETE. Call this with appropriate parameters to use the layer.


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
            builder = ControllerBuilder.new_builder(batch_dimension)
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
    def _is_act_not_complete(combined_state: Tuple[ACT_Controller, PyTree]) -> bool:
        controller, _ = combined_state
        return ~controller.is_completely_halted

    @staticmethod
    def _while_loop_adapter_factory(layer: '_ACTValidationWrapper'
                                    ) -> Callable[
        [Tuple[ACT_Controller, PyTree]],
        Tuple[ACT_Controller, PyTree]
    ]:
        """
        This is primarily an interface between the
        jax.lax.while_loop restrictions and how
        items are passed around to users.
        """

        def run_layer_adapter(state: Tuple[ACT_Controller, PyTree]
                              ) -> Tuple[ACT_Controller, PyTree]:
            controller, state = state
            controller, state = layer.run_layer(controller, state)
            return controller, state

        return run_layer_adapter
    @abstractmethod
    def make_controller(self,
                        initial_state: PyTree,
                        *args,
                        **kwargs
                        ) -> ACT_Controller:
        """

        A piece of the contract protocol.

        This method is responsible for creating a controller
        from the initial state (useful for batch shape information),
        and from any user-defined args or kwargs.

        You should use this section to:
            * Create a builder
            * Use the builder to define all accumulators you will need in your act computation
            * Build the controller.
            * Return the controller

        You will be detected violating the contract by the executing function if you:
            * Do not return a controller
            * Return a controller that is empty (has no accumulators defined).

        :param initial_state: The initial state, as seen by the act process.
                              This may be a jax PyTree if you need to pass a dictionary
                              or something around. See jax.tree_util for more info
        :param args: Any argument flags
        :param kwargs: Any keyword arg flags
        :return: An ACT_Controller instance with one or more accumulators.
        """
        raise NotImplementedError()

    @abstractmethod
    def run_iteration(self,
                  controller: ACT_Controller,
                  state: PyTree,
                  *args,
                  **kwargs,
                  ) -> Tuple[ACT_Controller, PyTree]:
        """
        A piece of the contract protocol that must be defined.

        This method is responsible for actually running a single
        act computation step. This should mean it:

        1) Accepts the current controller and state
        2) Updates the current state
        3) Uses the state to update the accumulators on the controller using cache_update
        4) Uses the state to produce the halting probabilities then commit them using iterate_act

        The new controller, and new state, should then be returned. This will be detected
        as violating its contract if:

            * A return is not a controller and state
            * A returned controller is the original controller instance.
            * A returned controller never had all its cached updates committed using iterate_act

        Additionally, you are also violating your contract, but will not be told explictly such
        by the class, if you:
            * Do not use the created controller correctly in run_iteration

        :param controller: The act controller for the current iteration
        :param state: The state for the current iteration.
        """
        raise NotImplementedError()

    def execute_act(self,
                    initial_state: PyTree,
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
        :param check_for_errors: Whether to check for errors or charge ahead blindly.
                                 Disabling check_for_errors should remove all errors
                                 overhead when compiled.
        :param *args: Any args to pass into make_controller
        :param **kwargs: Any keyword args to pass into make_controller.
        :return: A controller with the results stored in it, and the final state.
        """
        act_layer = _ACTValidationWrapper(self, check_for_errors)
        controller = act_layer.make_controller(initial_state, *args, **kwargs)

        wrapped_state = (controller, initial_state)
        run_layer = self._while_loop_adapter_factory(act_layer)
        controller, final_state = jax.lax.while_loop(self._is_act_not_complete,
                                                     run_layer,
                                                     wrapped_state
                                                     )
        return controller, final_state


class _ACTValidationWrapper:
    """
    Provides validation on returns.
    """
    def __init__(self,
                 layer: AbstractLayerMixin,
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

    ## Validation helpers. Checks for one particular issue
    @staticmethod
    def _validate_is_controller(item: Any, context: str):
        if not isinstance(item, ACT_Controller):
            msg = f"""
            Expected to be given a result of type {ACT_Controller}. 
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
            Expected tuple collection to have length of two. Instead,
            it had length of {len(item)}
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
            
            This is likely to occur if you did not properly account
            for the immutable nature of the controller when programming
            your layer. Did you use code that explictly reassigns variables
            like this?
            
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
            updates. You must commit your updates using iterate_act. Also
            make sure you are explicitly reassigning your variables, like follows:
            
            controller = controller.iterate_act(halting_probabilities) 
            
            rather than:
            
            controller.iterate_act(halting_probabilities)
            """
            msg = utils.format_error_message(msg, context)
            raise RuntimeError(msg)

        if controller.has_cached_updates:
            msg = f"""
            Controller had uncommitted updates and is not ready to commit. Make sure
            you use controller.cache_update to cache an update for every accumulator you defined.
            Also, make sure you use the immutable syntax where you explictly reassign variables:
            
            controller = controller.cache_update(...your update)
            
            Rather than:
            
            controller.cache_update(...your_update)
            """
            msg = utils.format_error_message(msg, context)
            raise RuntimeError(msg)


    def make_controller(self, initial_state: PyTree, *args, **kwargs) -> ACT_Controller:
        controller = self.layer.make_controller(initial_state, *args, **kwargs)
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

        update = self.layer.run_iteration(controller, state)
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
