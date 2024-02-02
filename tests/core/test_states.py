import unittest

from src.jax_act.core.states import ACTStates
import jax
import jax.numpy as jnp


class test_ACTState(unittest.TestCase):
    """
    Test that the act state operates sanely when Jitted
    """
    def test_can_maybe_jit(self):
        x = jnp.array([1, 2, 3])

        def make_act(x)->ACTStates:
            act_state = ACTStates(0.001,
                                  jnp.array([0, 0], dtype=jnp.int32),
                                  jnp.array([0.0, 0.0]),
                                  jnp.array([0.0, 0.0]),
                                  {},
                                  {},
                                  {},
                                  1.0

            )
            return act_state
        jax.config.update("jax_traceback_filtering", "off")

        jit_make_act = jax.jit(make_act)
        state = jit_make_act(x)
        print(state)
    def test_jit_complex(self):

        # Test we can create
        def make_act_state()->ACTStates:
            act_state = ACTStates(0.001,
                                  jnp.array([0, 0]),
                                  jnp.array([0.0, 0.0]),
                                  jnp.array([0.0, 0.0]),
                                  {"item" : jnp.array([0.0, 0.2])},
                                  {"item" : jnp.array([0.0, 0.3])},
                                  {"item" : None}

            )
            return act_state

        jit_state = jax.jit(make_act_state)

        # Test we can flatten

        flat_state, tree_def = jax.tree_util.tree_flatten(jit_state)
        state = jax.tree_util.tree_unflatten(tree_def, flat_state)

