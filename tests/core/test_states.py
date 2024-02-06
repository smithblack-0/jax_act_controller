import unittest

from src.jax_act.core.states import ACTStates
import jax
import jax.numpy as jnp
from jax.experimental import checkify

SHOW_ERROR_MESSAGES = True

class test_ACTState(unittest.TestCase):
    """
    Test that the act state operates sanely when Jitted
    """
    def resolve_checkification(self, function):
        checkified_function = checkify.checkify(function)
        jitted_function = jax.jit(checkified_function)
        def wrapper(*args, **kwargs):
            errors, output = jitted_function(*args, **kwargs)
            errors.throw()
            return output
        return wrapper
    def execute_with_checkification(self, uncheckified_function):
        checkified_function = checkify.checkify(uncheckified_function)
        jitted_function = jax.jit(checkified_function)
        errors, out = jitted_function()
        errors.throw()
        return out

    def make_state_mockup(self) -> ACTStates:
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
    def test_replace_epsilon(self):
        """
        Test that epsilon is confined to be between
        0 and 1
        """
        good_replacement = 0.6
        bad_replacement_too_low = -0.1
        bad_replacement_too_high = 1.2

        def test_function(replacement):
            state = self.make_state_mockup()
            return state.replace_epsilon(replacement)

        test_function = self.resolve_checkification(test_function)
        test_function = self.resolve_checkification(test_function)
        output = test_function(good_replacement)

        print(output)

        with self.assertRaises(ValueError) as err:
            test_function(bad_replacement_too_low)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)

        with self.assertRaises(ValueError) as err:
            test_function(bad_replacement_too_high)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)



    def test_can_jit(self):
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

