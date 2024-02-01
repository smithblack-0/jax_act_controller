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
from src.jax_act.frameworks.flax import AbstractFlaxACTLayer
jax.config.update("jax_traceback_filtering", "off")

class test_with_flax(unittest.TestCase):
    """
    Test that the mixin works correctly when
    utilized in the flax framework
    """
    def test_simple_functioning_flax_act_runs(self):
        """
        Test that a act instance with layers and
        parameters will correctly run when
        only dealing with parameters as a complication.
        """
        class DirectACT(AbstractFlaxACTLayer):
            """
            The test layer. This will consist of an act
            mechanism.
            """
            # Hyperparameters
            state_dim: int
            output_dim: int
            max_iterations: int = 10
            # Setup. We cannot do inline setup
            def setup(self) -> None:
                self.project_input = nn.Dense(self.state_dim)
                self.project_state = nn.Dense(self.state_dim)
                self.state_layernorm = nn.LayerNorm(self.state_dim)
                self.project_output = nn.Dense(self.output_dim)
                self.project_probs = nn.Dense(1)
            def update_state(self, state: jnp.ndarray)->jnp.ndarray:
                """
                Updates the state for the current iteration
                :param state: The current state
                :return: The new state
                """
                new_state = self.project_state(state)
                state = self.state_layernorm(new_state + state)
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

            # Fufill contract
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

            def make_controller(self,
                                initial_state: jnp.ndarray,
                                *args,
                                **kwargs)->ACT_Controller:
                batch_size = initial_state.shape[0]

                builder = jax_act.new_builder(batch_size)
                builder = builder.define_accumulator_by_shape("state", [batch_size, self.state_dim])
                builder = builder.define_accumulator_by_shape("output", [batch_size, self.output_dim])
                return builder.build()

            # Define main call
            @nn.compact
            def __call__(self, input: jnp.ndarray)->jnp.ndarray:
                initial_state = self.project_input(input)
                controller, state = self.execute_act(initial_state,
                                                     max_iterations=self.max_iterations)
                return controller["output"]

        batch_size = 16
        input_dim = 10
        state_dim = 20
        output_dim = 8

        rng = random.key(0)
        key_case, key_retained = random.split(rng)
        mock_data = 1.0*jnp.ones([batch_size, input_dim])
        layer_instance = DirectACT(
                                    state_dim = state_dim,
                                    output_dim = output_dim
                                    )
        parameters = layer_instance.init(key_case, mock_data)
        output = layer_instance.apply(parameters, mock_data)
        print(output)
    def test_stateful_act_runs(self):
        """
        Test that an act layer that has more complexities,
        such as state information for a running batch
        norm, runs
            """

        class StatefulACT(AbstractFlaxACTLayer):
            """
            The test layer. This will consist of an act
            mechanism.
            """
            # Hyperparameters
            state_dim: int
            output_dim: int

            # Setup. We cannot do inline setup
            def setup(self) -> None:
                self.project_input = nn.Dense(self.state_dim)
                self.project_state = nn.Dense(self.state_dim)
                self.project_output = nn.Dense(self.output_dim)
                self.project_probs = nn.Dense(1)
                self.batch_norm = nn.BatchNorm(use_running_average=True)
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

            # Fufill contract
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

            def make_controller(self,
                                initial_state: jnp.ndarray,
                                *args,
                                **kwargs)->ACT_Controller:
                batch_size = initial_state.shape[0]

                builder = jax_act.new_builder(batch_size)
                builder = builder.define_accumulator_by_shape("state", [batch_size, self.state_dim])
                builder = builder.define_accumulator_by_shape("output", [batch_size, self.output_dim])
                return builder.build()

            # Define main call
            @nn.compact
            def __call__(self, input: jnp.ndarray)->jnp.ndarray:
                initial_state = self.project_input(input)
                controller, _ = self.execute_act(initial_state)
                output = controller["output"]
                return self.batch_norm(output)


        batch_size = 16
        input_dim = 10
        state_dim = 20
        output_dim = 8

        rng = random.key(0)
        key_case, key_retained = random.split(rng)
        mock_data = jnp.zeros([batch_size, input_dim])
        layer_instance = StatefulACT(
                                    state_dim = state_dim,
                                    output_dim = output_dim
                                    )
        parameters = layer_instance.init(key_case, mock_data)
        output = layer_instance.apply(parameters, mock_data)



