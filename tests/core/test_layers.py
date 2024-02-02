"""

Unit tests for the framework-agnostic
layers mechanism

This is a fairly late stage set of tests, and
uses components from multiple locations like
builders rather than mocking up the features

As a result, it should be a later priority
to fix if everything suddenly breaks.


"""
import unittest
import jax
from jax import numpy as jnp
from src.jax_act.builder import ControllerBuilder
from src.jax_act.controller import ACT_Controller
from src.jax_act.layers import _ACTValidationWrapper, AbstractACTTemplate
from src.jax_act.types import PyTree
from typing import Tuple

SHOW_ERROR_MESSAGES = True
class testValidation(unittest.TestCase):
    """
    Tests to see if the validation wrapper
    is performing sanely

    As with most functions, we test jittability as we go along.
    While in some places we use staticargnums, they are never
    on tensors but on messages that would be compiled away anyhow
    when integrated.
    """
    def test_execute_validation(self):
        """ Test whether execute validation takes error state into account"""

        should_execute_instance = _ACTValidationWrapper(None, True)
        should_not_execute_instance = _ACTValidationWrapper(None, False)
        def error_causer():
            raise RuntimeError()

        # Test we execute when true
        with self.assertRaises(RuntimeError):
            should_execute_instance._execute_validation(error_causer)

        # Test we do not execute when suppressed
        should_not_execute_instance._execute_validation(error_causer)
    def test_validate_is_controller(self):

        builder = ControllerBuilder.new_builder([3, 5])
        controller = builder.build()
        context_message = "Message produced while testing validate_is_controller"
        test_function = _ACTValidationWrapper._validate_is_controller

        test_function(controller, context_message)
        jit_function = jax.jit(test_function, static_argnums=[1])
        jit_function(controller, context_message)

        with self.assertRaises(TypeError) as err:
            test_function(None, context_message)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)

    def test_validate_controller_not_empty(self):
        builder = ControllerBuilder.new_builder([3, 5])
        empty_controller = builder.build()

        builder = builder.define_accumulator_by_shape("test", [3, 5, 10])
        non_empty_controller = builder.build()

        context_string = "Message produced while testing validate_controller_not_empty"
        test_function = _ACTValidationWrapper._validate_controller_not_empty
        jit_function = jax.jit(test_function, static_argnums=[1])

        test_function(non_empty_controller, context_string)
        jit_function(non_empty_controller, context_string)

        with self.assertRaises(RuntimeError) as err:
            test_function(empty_controller, context_string)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)

    def test_validate_is_tuple(self):

        tuple = (3, 5)
        nontuple = "test"
        context_string = "Message produced while testing validate_controller_not_empty"
        target_function = _ACTValidationWrapper._validate_is_tuple
        jit_function = jax.jit(target_function, static_argnums=[1])

        target_function(tuple, context_string)
        jit_function(tuple, context_string)
        with self.assertRaises(TypeError) as err:
            target_function(nontuple, context_string)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)

    def test_validate_length_two(self):

        two_length = (1, 2)
        three_length = (1, 2, 3)
        context_string = "Message produced while testing validate_length_two"
        target_function = _ACTValidationWrapper._validate_length_two
        jit_function = jax.jit(target_function, static_argnums=[1])

        target_function(two_length, context_string)
        jit_function(two_length, context_string)

        with self.assertRaises(RuntimeError) as err:
            target_function(three_length, context_string)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)

    def test_validate_is_novel_controller(self):
        builder = ControllerBuilder.new_builder([3, 5])
        first_controller = builder.build()
        second_controller = builder.build()
        context_string = "Message produced while testing validate_is_novel_controller"

        target_function = _ACTValidationWrapper._validate_is_novel_controller
        jit_function = jax.jit(target_function, static_argnums=[2])

        target_function(first_controller, second_controller, context_string)
        jit_function(first_controller, second_controller, context_string)

        with self.assertRaises(RuntimeError) as err:
            target_function(first_controller,
                                                                first_controller,
                                                                context_string)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)
    def test_validate_all_act_updates_committed(self):

        builder = ControllerBuilder.new_builder([3])
        builder = builder.define_accumulator_by_shape("test_accumulator_one", [3, 7])
        builder = builder.define_accumulator_by_shape("test_accumulator_two", [3])
        controller = builder.build()

        controller_needs_more_updates = controller.cache_update("test_accumulator_one", jnp.ones([3, 7]))
        controller_needs_halting_probabilities = controller_needs_more_updates.cache_update("test_accumulator_two",
                                                                                            jnp.ones([3]))
        context_string = "Message produced while testing validate_all_act_updates_committed"

        target_function = _ACTValidationWrapper._validate_all_act_updates_committed
        jit_function = jax.jit(target_function, static_argnums=[1])

        target_function(controller, context_string)
        jit_function(controller, context_string)

        with self.assertRaises(RuntimeError) as err:
            target_function(controller_needs_more_updates, context_string)
        if SHOW_ERROR_MESSAGES:
            print("Testing error message when more caching is needed")
            print(err.exception)

        with self.assertRaises(RuntimeError) as err:
            target_function(controller_needs_halting_probabilities, context_string)
        if SHOW_ERROR_MESSAGES:
            print("Testing error message when need to call iterate_act")
            print(err.exception)

    #TODO: Consider testing whether make_controller and run_layer wrappers throw
    # when hitting error cases .
    #
    # I am skipping this for now because the code is extremely easy to inspect by
    # eye, and the methods being called are validated above.

    def test_make_controller(self):
        """ Test a simple make controller case"""
        class ValidACT(AbstractACTTemplate):
            def make_controller(self, state: jnp.ndarray, *args, **kwargs)->ACT_Controller:
                batch_shape = state.shape[0]
                builder = ControllerBuilder.new_builder(batch_shape)
                builder = builder.define_accumulator_by_shape("test", [batch_shape, 2])
                return builder.build()

            def setup_state(self, controller: ACT_Controller, state: PyTree):
                pass
            def run_iteration(self,
                          controller: ACT_Controller,
                          state: PyTree,
                          *args,
                          **kwargs
                          ) -> Tuple[ACT_Controller, PyTree]:
                # This is a mockup.
                return controller, state

        valid_layer = ValidACT()
        valid_instance = _ACTValidationWrapper(valid_layer, True)
        input = jnp.ones([7, 2])
        controller = valid_instance.make_controller(input)
        self.assertIsInstance(controller, ACT_Controller)

    def test_make_controller_using_arguments(self):
        """Test make controller works when passing user flags"""
        class ValidACT(AbstractACTTemplate):
            def make_controller(self, state: jnp.ndarray, length: int)->ACT_Controller:
                batch_shape = list(state.shape[0:length])
                builder = ControllerBuilder.new_builder(batch_shape)
                builder = builder.define_accumulator_by_shape("test", [*batch_shape, 7, 3])
                return builder.build()
            def setup_state(self, controller: ACT_Controller, state: PyTree):
                pass
            def run_iteration(self,
                          controller: ACT_Controller,
                          state: PyTree,
                          ) -> Tuple[ACT_Controller, PyTree]:
                # This is a mockup.
                return controller, state

        valid_make_protocol = ValidACT()
        valid_instance = _ACTValidationWrapper(valid_make_protocol, True)

        input = jnp.ones([7, 3])
        length = 2

        controller = valid_instance.make_controller(input, length)
        controller = valid_instance.make_controller(input, length=length)

        self.assertIsInstance(controller, ACT_Controller)
    def test_run_layer(self):

        class ValidACT(AbstractACTTemplate):
            def make_controller(self, state: jnp.ndarray)->ACT_Controller:
                batch_shape = state.shape[0]
                builder = ControllerBuilder.new_builder(batch_shape)
                builder = builder.define_accumulator_by_shape("test", [batch_shape, 2])
                return builder.build()
            def setup_state(self, controller: ACT_Controller, state: PyTree):
                pass
            def run_iteration(self,
                          controller: ACT_Controller,
                          state: PyTree,
                          ) -> Tuple[ACT_Controller, PyTree]:
                batch_shape = state.shape[0]

                controller = controller.cache_update("test", jnp.ones([batch_shape, 2]))

                halting_probs = 0.5*jnp.ones([batch_shape])
                controller = controller.iterate_act(halting_probs)
                return controller, state

        valid_instance = ValidACT()
        valid_instance = _ACTValidationWrapper(valid_instance, True)

        state = jnp.ones([3])
        controller = valid_instance.make_controller(state)
        self.assertIsInstance(controller, ACT_Controller)

class test_AbstractLayerMixin(unittest.TestCase):
    """
    Test that the abstract layer mixin
    can be reasonably used to perform
    the various tasks that may be demanded of it
    """
    class ACTACT(AbstractACTTemplate):
        """
        A pet test layer for testing
        that the mixin functions properly
        when it is the only update.
        """
        def update_state(self, state: jnp.ndarray) -> jnp.ndarray:
            # Mock function
            return state + 0.1*jnp.ones_like(state)
        def make_probabilities(self, state: jnp.ndarray) -> jnp.ndarray:
            # Mock function
            batch_shape = state.shape[0]
            return 0.1 * jnp.ones([batch_shape])
        def setup_state(self, controller: ACT_Controller, state: PyTree):
            pass
        def make_output(self, state: jnp.ndarray) -> jnp.ndarray:
            # Mock function
            batch_shape = state.shape[0]
            return 0.1 * jnp.ones([batch_shape, self.embedding_dim])

        def make_controller(self, state: jnp.ndarray, *args, **kwargs) -> ACT_Controller:
            batch_shape = state.shape[0]
            builder = ControllerBuilder.new_builder(batch_shape)
            builder = builder.define_accumulator_by_shape("state", list(state.shape))
            builder = builder.define_accumulator_by_shape("output", [batch_shape, self.embedding_dim])
            return builder.build()

        def run_iteration(self,
                          controller: ACT_Controller,
                          state: jnp.ndarray,
                          *args,
                          **kwargs) -> Tuple[ACT_Controller, jnp.ndarray]:

            state = self.update_state(state)
            output = self.make_output(state)
            probs = self.make_probabilities(state)

            controller = controller.cache_update("state", state)
            controller = controller.cache_update("output", output)
            controller = controller.iterate_act(probs)

            return controller, state
        def __call__(self, state: jnp.ndarray):
            return self.execute_act(state)
        def __init__(self, embedding_dim: int):
            self.embedding_dim = embedding_dim
    def test_act(self):
        embedding_dim = 10
        layer = self.ACTACT(embedding_dim)
        initial_state = jnp.zeros([7])
        controller, state = layer(initial_state)

