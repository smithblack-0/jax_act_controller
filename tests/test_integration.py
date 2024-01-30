"""
A location for integration testing

TODO:
    * Framework integration tests for layers
    *
"""
import unittest
from typing import Tuple

import jax.lax
import numpy as np

from jax import numpy as jnp
from src import jax_act
class test_act_processes(unittest.TestCase):
    """
    Test that some act processes are easily
    handled using the provided features.

    These are run in standard mode, without
    any compilation or jit mechanisms.
    """

    def test_basic_act(self):
        """ Test the basic ACT process"""

        batch_shape = 10
        state_shape = [batch_shape, 20]
        output_shape = [batch_shape, 10]

        def mockup_halting_probabilities()->np.ndarray:
            return 0.1*jnp.ones([batch_shape])

        def mockup_state_update()->np.ndarray:
            return 0.1*jnp.ones(state_shape)

        def mockup_output_update()->np.ndarray:
            return 0.1*jnp.ones(output_shape)

        # Define the ACT process
        builder = jax_act.ControllerBuilder.new_builder(batch_shape)
        builder = builder.define_accumulator_by_shape("state", state_shape)
        builder = builder.define_accumulator_by_shape("output", output_shape)
        controller = builder.build()

        # Perform ACT process
        while ~controller.is_completely_halted:

            new_state = mockup_state_update()
            new_output = mockup_output_update()
            new_probabilities = mockup_halting_probabilities()

            controller = controller.cache_update("state", new_state)
            controller = controller.cache_update("output", new_output)
            controller = controller.iterate_act(new_probabilities)

class test_functional_act_processes(unittest.TestCase):
    def test_basic_act_functional(self):
        """
        Test that we can perform a vanilla act
        loop as laid out in the paper while operating
        in functional mode. This means using jax.lax.while_loop.
        """

        batch_shape = 10
        state_shape = [batch_shape, 20]
        output_shape = [batch_shape, 10]

        # Create process mockups
        def mockup_halting_probabilities()->np.ndarray:
            return 0.1*jnp.ones([batch_shape])

        def mockup_state_update()->np.ndarray:
            return 0.1*jnp.ones(state_shape)

        def mockup_output_update()->np.ndarray:
            return 0.1*jnp.ones(output_shape)

        # Create builder and initial state functions

        def build_controller()->jax_act.ACT_Controller:
            builder = jax_act.ControllerBuilder.new_builder(batch_shape)
            builder = builder.define_accumulator_by_shape("state", state_shape)
            builder = builder.define_accumulator_by_shape("output", output_shape)
            return builder.build()

        # Define internal type
        act_current_state = Tuple[jax_act.ACT_Controller, jnp.ndarray]

        def build_initial()->act_current_state:
            initial_state = mockup_state_update()
            initial_controller = build_controller()
            return (initial_controller, initial_state)

        def check_if_done(act_state: act_current_state)->bool:
            controller, _ = act_state
            #Note: It is VERY important the following logical negation is
            # not performed using the "not" operator
            return ~controller.is_completely_halted

        def run_act_process(act_state: act_current_state):
            controller, state = act_state

            new_state = mockup_state_update()
            new_output = mockup_output_update()
            new_probabilities = mockup_halting_probabilities()

            controller = controller.cache_update("state", new_state)
            controller = controller.cache_update("output", new_output)
            controller = controller.iterate_act(new_probabilities)

            return (controller, new_state)


        # Standard loop. For comparison purposes.
        internal_state = build_initial()
        while check_if_done(internal_state):
            internal_state = run_act_process(internal_state)
        normal_controller, _ = internal_state

        # Functional loop.
        act_initial_state = build_initial()
        act_final_state = jax.lax.while_loop(check_if_done, run_act_process, act_initial_state)
        functional_controller, _ = act_final_state

        original_leafs, _ = jax.tree_util.tree_flatten(normal_controller)
        functional_leafs, _ = jax.tree_util.tree_flatten(functional_controller)

        # Handle the epsilon separately.
        #
        # Unfortunately, the types are different, so we have to manually
        # pull it out and compare

        original_epsilon = original_leafs[0]
        functional_epsilon = functional_leafs[0]
        self.assertTrue(original_epsilon == functional_epsilon)

        # Handle the array checks

        original_leafs = original_leafs[1:]
        functional_leafs = functional_leafs[1:]
        self.assertTrue(jax_act.utils.are_pytrees_equal(original_leafs, functional_leafs))


class test_execute_act(unittest.TestCase):
    """
    Test execute_act.
    """

    class ACTMockLayer:

        def update_state(self, state: jnp.ndarray) -> jnp.ndarray:
            # Mock function
            return state + 0.1 * jnp.ones_like(state)

        def make_probabilities(self, state: jnp.ndarray) -> jnp.ndarray:
            # Mock function
            batch_shape = state.shape[0]
            return 0.1 * jnp.ones([batch_shape])

        def make_output(self, state: jnp.ndarray) -> jnp.ndarray:
            # Mock function
            batch_shape = state.shape[0]
            return 0.1 * jnp.ones([batch_shape, self.embedding_dim])

        def make_controller(self, state: jnp.ndarray) -> jax_act.ACT_Controller:
            batch_shape = state.shape[0]
            builder = jax_act.ControllerBuilder.new_builder(batch_shape)
            builder = builder.define_accumulator_by_shape("state", list(state.shape))
            builder = builder.define_accumulator_by_shape("output", [batch_shape, self.embedding_dim])
            return builder.build()

        def run_layer(self, controller: jax_act.ACT_Controller, state: jnp.ndarray
                      ) -> Tuple[jax_act.ACT_Controller, jnp.ndarray]:
            state = self.update_state(state)
            output = self.make_output(state)
            probs = self.make_probabilities(state)

            controller = controller.cache_update("state", state)
            controller = controller.cache_update("output", output)
            controller = controller.iterate_act(probs)

            return controller, state

        def __init__(self,
                     embedding_dim: int = 10,
                     ):
            self.embedding_dim = embedding_dim

    def get_layer(self, embedding_dim: int) -> 'ACTMockLayer':
        return self.ACTMockLayer(embedding_dim)

    def test_execute_act(self):
        """ Test that a valid instance of execute act will work"""
        batch_dim = 10
        embedding_dim = 12

        initial_state = jnp.zeros([batch_dim, embedding_dim])
        layer = self.get_layer(embedding_dim)

        controller, state = jax_act.execute_act(layer,
                                        initial_state)
        self.assertTrue(jnp.all(initial_state != state))

    def test_execute_act_jit(self):
        """ Test that a valid instance of execute act will work under jit"""
        batch_dim = 10
        embedding_dim = 12

        initial_state = jnp.zeros([batch_dim, embedding_dim])
        layer = self.get_layer(embedding_dim)

        def execute_act_jit_helper(state):
            # Jit does not like passing around layers
            #
            # However, as users are usually going to be passing
            # in the layers from self, it should be fine.
            return jax_act.execute_act(layer, state)

        jit_function = jax.jit(execute_act_jit_helper)
        controller, state = jit_function(initial_state)
        self.assertTrue(jnp.all(initial_state != state))