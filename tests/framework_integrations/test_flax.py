import flax
import jax
import optax
import unittest

from flax import linen as nn
from jax import numpy as jnp
from jax import random
from flax.training import train_state
from typing import Tuple, Callable

from src import jax_act
from src.jax_act import ACT_Controller, ControllerBuilder, AbstractLayerMixin, PyTree

jax.config.update("jax_traceback_filtering", "off")

class test_with_flax(unittest.TestCase):
    """
    Test that the mixin works correctly when
    utilized in the flax framework
    """
    class SimpleACTLayer(AbstractLayerMixin, nn.Module):
        """
        The test layer. This will consist of a simple act
        mechanism that has a single batch dimension.

        It accepts an incoming dimension, uses it
        to
        """
        # Hyperparameters
        state_dim: int
        output_dim: int

        def setup(self) -> None:
            self.project_input = nn.Dense(self.state_dim)
            self.project_state = nn.Dense(self.state_dim)
            self.project_output = nn.Dense(self.output_dim)
            self.project_probs = nn.Dense(1)
        def make_controller(self,
                            initial_state: jnp.ndarray,
                            *args,
                            **kwargs)->ACT_Controller:
            batch_size = initial_state.shape[0]

            builder = jax_act.new_builder(batch_size)
            builder = builder.define_accumulator_by_shape("state", [batch_size, self.state_dim])
            builder = builder.define_accumulator_by_shape("output", [batch_size, self.output_dim])
            return builder.build()

        def update_state(self, state: jnp.ndarray)->jnp.ndarray:
            """
            Updates the state for the current iteration
            :param state: The current state
            :return: The new state
            """
            new_state = self.project_state(state)
            state = nn.relu(new_state + state)
            return state

        def make_output(self, state: jnp.ndarray)->jnp.ndarray:
            output = self.project_output(state)
            output = nn.relu(output)
            return output

        def make_halting_probabilities(self, state: jnp.ndarray)->jnp.ndarray:
            logits = self.project_probs(state)
            probabilities = nn.sigmoid(logits)
            probabilities = jnp.squeeze(probabilities, axis=-1)
            return probabilities

        def _is_act_not_complete(self, combined_state: Tuple[ACT_Controller, PyTree]) -> bool:
            controller, _ = combined_state

        def run_iteration(self,
                  controller: ACT_Controller,
                  state: jax_act.PyTree,
                  *args,
                  **kwargs,
                  ) -> Tuple[ACT_Controller, jax_act.PyTree]:

            state = self.update_state(state)
            output = self.make_output(state)
            halting_probabilities = self.make_halting_probabilities(state)

            controller = controller.cache_update("state", state)
            controller = controller.cache_update("output", output)
            controller = controller.iterate_act(halting_probabilities)

            return controller, state
        def loop_adapter(self, state: Tuple[ACT_Controller, PyTree])->Tuple[ACT_Controller,PyTree]:
            controller, state = state
            controller, state = self.run_iteration(controller, state)
            return controller, state
        @nn.compact
        def __call__(self, input: jnp.ndarray)->jnp.ndarray:
            initial_state = self.project_input(input)
            controller = self.make_controller(initial_state)
            combined_state = (controller, initial_state)

            def conditional_predicate(self, state: Tuple[ACT_Controller, jnp.ndarray])->bool:
                controller, _ = state
                return ~controller.is_completely_halted

            if self.is_mutable_collection('params'):
                combined_state = self.loop_adapter(combined_state)

            output = nn.while_loop(conditional_predicate,
                                   self.loop_adapter,
                                   self,
                                   combined_state,
                                   broadcast_variables="params"
                                   )
            return controller["output"]

    def test_running_act_layer(self):
        """ Test that the act layer can be run"""
        batch_size = 16
        input_dim = 10
        state_dim = 20
        output_dim = 8

        rng = random.key(0)
        key_case, key_retained = random.split(rng)
        mock_data = jnp.zeros([batch_size, input_dim])
        layer_instance = self.SimpleACTLayer(
                                    state_dim = state_dim,
                                    output_dim = output_dim)
        parameters = layer_instance.init(key_case, mock_data)
        output = layer_instance.apply(parameters, mock_data)






