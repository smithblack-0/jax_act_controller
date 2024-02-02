from abc import ABC, abstractmethod
from typing import Tuple, Union, List, Optional, Callable, Any

from jax import numpy as jnp
from src.jax_act.core.controller import PyTree, ACT_Controller
from src.jax_act.core.builder import ControllerBuilder
from src.jax_act.core import utils


class AbstractControllerContract(ABC):
    """
    The controller interface definiton.

    This is an abstract contract for a
    Probability Driven Accumulator of some sort.

    Certain mechanisms are needed to interface
    with the controller properly, including knowing
    how to make one and knowing what to store in one

    This presents abstract methods and validation functions
    for this functionality

    ------ methods ------

    Abstract:
    * make_controller: returns a valid controller instance
    * run_iteration: Runs a single act layer, and caches then commits the accumulated features.
    * setup_state:  Can be used to handle any state details.

    Utility:

    new_builder: Returns a new builder. Use it in make controller
    """

    def new_builder(cls,
                    batch_shape: Union[int, List[int]],
                    core_dtype: Optional[jnp.dtype] = None,
                    epsilon: float = 1e-4,
                    depression_constant: float = 1.0
                    )->'ControllerBuilder':
        """
        Creates a new builder you can then
        edit.

        :param batch_shape: The batch shape. Can be an int, or a list of ints
        :param core_dtype: The dtype of data
        :param epsilon: The epsilon for the act threshold. Note that if you are passing this
                        as a variable, it will need to be a staticargnum when compiled.
        :param depression_constant: This multiplies the halting probability. It can used with
                        depression scheduling to encourage longer sequence exploration.
        :return: A Builder instance
        """
        return ControllerBuilder.new_builder(batch_shape,
                                             core_dtype,
                                             epsilon,
                                             depression_constant
                                             )
    @abstractmethod
    def make_controller(self,
                        initial_state: PyTree,
                        depression_constant: float,
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
    def setup_lazy_parameters(self, controller: ACT_Controller, state: PyTree):
        """
        A piece of the contract protocol

        Some frameworks, such as flax, rely on lazy instancing
        of certain parameters. This can cause issues when entering
        a while loop. Running .run_iteration once when meeting
        the appropriate condition fixes this

        Room is left for this in this location. Either execute
        run_iteration once if you meet the criteria, or fill
        it with a 'pass' statement if you do not need it.
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


class ContractValidationWrapper:
    """
    Provides validation on returns.
    """
    def __init__(self,
                 layer: AbstractControllerContract,
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

    def run_iteration(self,
                      controller: ACT_Controller,
                      state: PyTree
                      ) -> Tuple[ACT_Controller, PyTree]:

        update = self.layer.run_iteration(controller, state)

        def validate():
            # Perform return formatting validation
            context_message = "An issue occurred while validating the execution of run_iteration"
            self._validate_is_tuple(update, context_message)
            self._validate_length_two(update, context_message)

            # Validate the controller
            new_controller, _ = update
            self._validate_is_controller(new_controller, context_message)
            self._validate_is_novel_controller(controller, new_controller, context_message)
            self._validate_all_act_updates_committed(controller, context_message)

        self._execute_validation(validate)
        return update
