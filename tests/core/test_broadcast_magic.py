import unittest
import jax
from jax import numpy as jnp
from src.jax_act.core.broadcast_editor import BroadcastEditor
from src.jax_act.core.states import ACTStates
SHOW_ERROR_MESSAGES = True

def make_state_mockup()->ACTStates:
    epsilon = 0.1
    iterations = jnp.array([3, 3, 2])
    probabilities = jnp.array([0.1, 0.5, 1.0])
    residuals = jnp.array([0.0, 0.0, 0.3])
    accumulator = {"state": jnp.array([[3.0, 4.0, -1.0], [0.5, 1.2, 2.1], [0.7, 0.3, 0.5]]),
                   "output": jnp.array([[0.1, 0.1, 0.1], [1.0, 0.7, 0.3], [0.1, 0.1, 0.1]])
                   }
    update = {"state": jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [-1.0, 2.0, 1.0]]),
              "output": jnp.array([[0.1, 0.2, 0.2], [1.2, 1.2, 1.2], [0.1, 0.1, 0.1]])}
    return ACTStates(epsilon = epsilon,
                     iterations = iterations,
                     probabilities=probabilities,
                     residuals = residuals,
                     accumulators=accumulator,
                     defaults = accumulator,
                     updates=update,
                     depression_constant=1.0
                     )



class test_magic_methods(unittest.TestCase):
    """
    Test suite for the magic methods.
    """
    def test_addition(self):
        pass

