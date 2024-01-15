import unittest

from src.states import ACTStates, accumulation_state_flatten, accumulation_state_unflatten
import jax
import flax
import flax.linen as nn
import jax.numpy as jnp

from dataclasses import dataclass
from typing import Dict, Optional

class test_ACTState(unittest.TestCase):
    """
    Test that the act state operates sanely when Jitted
    """
    def test_can_jit_act_state(self):
        x = jnp.array([1, 2, 3])

        def make_act(x)->ACTStates:
            return ACTStates(1,
                                         x,
                                         x,
                                         {"test" : x},
                                         {"test" : x},
                                         {"test" : None}
                                         )
        jit_make_act = jax.jit(make_act)
        jit_make_act(x)

    def test_saveloadable(self):

        # Custom Flax layer using ACTState
        class ACTLayer(nn.Module):
            @nn.compact
            def __call__(self, x):
                # Initialize ACTState - replace with actual initialization
                act_state = ACTAccumulationStates(x,
                                                  x,
                                                  {"test" : x},
                                                  {"test" : x}
                                                  )
                return act_state


        # Test function for JIT compilation
        @jax.jit
        def test_jit(x, model):
            return model(x)


        # Initialize the model and input
        revised_layer = flax.linen.jit(ACTLayer)
        model = revised_layer()
        x = jnp.array([1, 2, 3])

        # Test saving and loading
        params = model.init(jax.random.PRNGKey(0), x)
        bytes_data = flax.serialization.to_bytes(params)
        with open('model_checkpoint', 'wb') as f:
            f.write(bytes_data)

        with open('model_checkpoint', 'rb') as f:
            bytes_data = f.read()
        loaded_params = flax.serialization.from_bytes(model.init(jax.random.PRNGKey(0), x), bytes_data)
        print("Model saved and loaded successfully")
        print(loaded_params)

