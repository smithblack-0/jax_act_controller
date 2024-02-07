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

class test_helper_functions(unittest.TestCase):
    def test_execute_binary_operator_simple(self):

        context_msg = "testing utility execute binary operator"
        # Test edge case: tensor with tensor
        operator = jnp.add
        operand_a = jnp.ones([3])
        operand_b = jnp.ones([3, 7])
        expected_result = operand_a[..., None] + operand_b
        result = BroadcastEditor.execute_binary_operator(operator, operand_a, operand_b, context_msg)
        result = BroadcastEditor.execute_binary_operator(operator, operand_b, operand_a, context_msg)
        self.assertTrue(jnp.allclose(result, expected_result))

    def test_execute_binary_operator_with_pytree(self):
        context_msg = "testing utility execute binary operator with a pytree"

        # Test tensor with pytree
        operator = jnp.add
        operand_a = jnp.ones([3])
        operand_b = {"item1" : jnp.ones([3, 5]), "item2" : jnp.ones([3, 7])}
        expected = {"item1" : 2*jnp.ones([3, 5]), "item2" : 2*jnp.ones([3, 7])}
        result = BroadcastEditor.execute_binary_operator(operator, operand_a, operand_b, context_msg)
        result = BroadcastEditor.execute_binary_operator(operator, operand_b, operand_a, context_msg)
        self.assertTrue(jnp.allclose(expected["item1"], result["item1"]))
        self.assertTrue(jnp.allclose(expected["item2"], result["item2"]))

    def test_execute_binary_operator_noncommutive(self):
        context_msg = "testing utility execute binary operator with noncommutive"

        operator = jnp.subtract
        operand_a = 2*jnp.ones([3])
        operand_b = jnp.ones([3, 5])
        expected_result = operand_a[..., None]-operand_b
        result = BroadcastEditor.execute_binary_operator(operator, operand_a, operand_b, context_msg)
        self.assertTrue(jnp.allclose(expected_result,result))

    def test_execute_binary_operator_jit(self):
        context_msg = "testing execute binary operator with noncommutive"

        operator = jnp.subtract
        operand_a = 2*jnp.ones([3])
        operand_b = jnp.ones([3, 5])
        expected_result = operand_a[..., None]-operand_b
        func = jax.jit(BroadcastEditor.execute_binary_operator, static_argnums=[3])
        result = func(jax.tree_util.Partial(operator), operand_a, operand_b, context_msg)
        self.assertTrue(jnp.allclose(expected_result,result))

    def test_execute_binary_operator_raises_utility_wrapper(self):
        context_msg = "testing execute binary operator raises and informs when the utility function breaks"
        tree_one = {"item1" : jnp.ones([3])}
        tree_two = {"item2" : jnp.ones([4])}

        with self.assertRaises(RuntimeError) as err:
            BroadcastEditor.execute_binary_operator(jnp.add, tree_one, tree_two, context_msg)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)
            print(err.exception.__cause__)

    def test_execute_binary_operator_raises_bad_operator(self):
        context_message = """
                          testing execute binary operator when
                          the operator is corrupt
                          """
        operand_one = jnp.ones([3, 4])
        operand_two = jnp.ones([2, 7])
        with self.assertRaises(RuntimeError) as err:
            BroadcastEditor.execute_binary_operator(lambda x : x, operand_one, operand_two,
                                                    context_message)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)
            print(err.exception.__cause__)

    def test_execute_unitary_operator(self):
        context_msg = "testing utility execute binary operator with noncommutive"

        operator = jnp.negative
        operand = jnp.ones([3, 2])
        expected_result = operator(operand)

    def test_execute_unitary_operator_with_pytree(self):

        context_msg = "testing unitary operator executer"
        operator = jnp.negative
        operand = {"item1" : jnp.ones([3, 5]), "item2" : jnp.ones([3, 7])}
        expected_output = {"item1" : -jnp.ones([3, 5]),
                           "item2" : -jnp.ones([3,7])}
        result = BroadcastEditor.execute_unitary_operation(operator, operand, context_msg)
        self.assertTrue(jnp.allclose(expected_output["item1"], result["item1"]))
        self.assertTrue(jnp.allclose(expected_output["item2"], result["item2"]))

    def test_execute_unitary_operator_raise_bad_operator(self):
        context_msg = "testing execute_unitary_operator with a bad operator"
        operator = lambda x, y : x
        operand = jnp.ones([3, 2])
        with self.assertRaises(RuntimeError) as err:
            BroadcastEditor.execute_unitary_operation(operator, operand, context_msg)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)
            print(err.exception.__cause__)


class test_magic_methods(unittest.TestCase):
    """
    Test suite for the magic methods.
    """
    def test_addition(self):
        pass

