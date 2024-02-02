import unittest
from typing import Tuple

from jax import numpy as jnp

from src.jax_act import ACT_Controller, PyTree, ControllerBuilder, AbstractACTTemplate


class test_AbstractLayerMixin(unittest.TestCase):
    """
    Test that the abstract layer mixin
    can be reasonably used to perform
    the various tasks that may be demanded of it
    """
    class ACT(AbstractACTTemplate):
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

        def make_output(self, state: jnp.ndarray) -> jnp.ndarray:
            # Mock function
            batch_shape = state.shape[0]
            return 0.1 * jnp.ones([batch_shape, self.embedding_dim])

        def make_controller(self, state: jnp.ndarray, *args, **kwargs) -> ACT_Controller:
            batch_shape = state.shape[0]
            builder = self.new_builder(batch_shape)
            builder = builder.define_accumulator_by_shape("state", list(state.shape))
            builder = builder.define_accumulator_by_shape("output", [batch_shape, self.embedding_dim])
            return builder.build()

        def setup_lazy_parameters(self, controller: ACT_Controller, state: PyTree):
            # No tricks needed, nothing is lazy
            pass
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
        layer = self.ACT(embedding_dim)
        initial_state = jnp.zeros([7])
        controller, state = layer(initial_state)
