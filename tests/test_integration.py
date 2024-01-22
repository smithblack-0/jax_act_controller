"""
A location for integration testing
"""
import unittest
from typing import Tuple

import jax.lax
import numpy as np

from jax import numpy as jnp
from src import jax_act



class test_common_dynamic_act_processes(unittest.TestCase):
    """
    Test that common act processes are easily
    handled using the provided features.
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
        while not controller.is_completely_halted:

            new_state = mockup_state_update()
            new_output = mockup_output_update()
            new_probabilities = mockup_halting_probabilities()

            controller = controller.cache_update("state", new_state)
            controller = controller.cache_update("output", new_output)
            controller = controller.iterate_act(new_probabilities)

        print(controller.iterations)
        print(controller.accumulators)



class test_functional_act_processes(unittest.TestCase):
    def test_basic_act(self):

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
            return not controller.is_completely_halted

        def run_act_process(act_state: act_current_state):
            controller, state = act_state

            new_state = mockup_state_update()
            new_output = mockup_output_update()
            new_probabilities = mockup_halting_probabilities()

            controller = controller.cache_update("state", new_state)
            controller = controller.cache_update("output", new_output)
            controller = controller.iterate_act(new_probabilities)

            return (controller, new_state)

        run_act_jit = jax.jit(run_act_process)

        # Raw loop
        internal_state = build_initial()
        while check_if_done(internal_state):
            internal_state = run_act_process(internal_state)

        controller, _ = internal_state
        print(controller.probabilities)

        # This loop

        act_current_state = build_initial()
        act_final_state = jax.lax.fori_loop(0, 10, lambda i, x : run_act_process(x), act_current_state)
        controller, _, = act_final_state
        print(controller.probabilities)

        #act_initial_state = build_initial()
        #act_final_state = jax.lax.while_loop(check_if_done, run_act_process, act_initial_state)

