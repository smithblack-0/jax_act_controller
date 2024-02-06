import unittest
from typing import Tuple, List

import jax.tree_util
from numpy import random

from src.jax_act.core.types import PyTree
from src.jax_act.core import utils
from jax import numpy as jnp
from src.jax_act.core.states import ACTStates


def make_empty_state_mockup() -> ACTStates:
    return ACTStates(
        epsilon=0,
        iterations=None,
        accumulators=None,
        defaults=None,
        updates=None,
        probabilities=None,
        residuals=None,
        depression_constant=1.0
    )

SHOW_ERROR_MESSAGES = True
class test_utilities(unittest.TestCase):
    """
    Test that the private helper functions used by the
    controller are acting sanely.

    We mock up data to represent a condition, then see if the
    helpers act appropriately to produce a new state
    """
    def test_setup_left_broadcast(self):
        """ Test the broadcast helper works properly"""
        item1 = random.randn(2, 5)
        item2 = random.randn(2, 5, 6, 8)
        item3 = random.randn(2)

        # Test same length works
        setup = utils.setup_left_broadcast(item1, item1)
        setup + item1

        # Test making it broader works
        setup = utils.setup_left_broadcast(item1, item2)
        item2 + setup

        #Test simple throw works
        with self.assertRaises(AssertionError):
            utils.setup_left_broadcast(item1, item3)
    def test_are_pytree_structure_equal(self):
        """
        Test to see if we can detect when pytrees are shaped similarly.
        """

        tree = {"item1" : random.randn(3, 5, 10, 2),
                "item2" : random.randn(2, 4, 5),
                "item3" : [random.randn(2, 4), random.randn(4, 5)]}
        tree2 = {"item1" : None, "item2" : None, "item3" : [None, None]}
        tree3 = {}

        # Test if we detect when trees have same structure
        self.assertTrue(utils.are_pytree_structure_equal(tree, tree))
        self.assertTrue(utils.are_pytree_structure_equal(tree, tree2))
        self.assertFalse(utils.are_pytree_structure_equal(tree, tree3))

    def test_merge_pytrees(self):
        """
        Test that the function responsible for pytree merging actually
        works as expected
        """

        tensor = jnp.array([0.0, 1.0, 1.2, 0.1])
        tensor2 = jnp.array([2.0, 1.0, 0.2, 0.1])

        # Test edge case: Only leafs

        operator = lambda a, b : a - b
        expected_tensor = jnp.array([-2.0, 0, 1.0, 0.0])
        outcome = utils.merge_pytrees(operator, tensor, tensor2)
        self.assertTrue(jnp.allclose(expected_tensor, outcome))

        # Test tree case.

        tree = {"item" : tensor, "item2" : tensor, "item3" : [tensor, tensor]}
        tree2 = {"item" : tensor2, "item2" : tensor2, "item3" : [tensor2, tensor2]}
        expected  = {"item" : expected_tensor, "item2" : expected_tensor,
                   "item3" : [expected_tensor, expected_tensor]}

        outcome = utils.merge_pytrees(operator, tree, tree2)
        self.assertTrue(jnp.allclose(outcome['item'], expected['item']))

class test_checkify_wrapper(unittest.TestCase):
    """
    The checkify wrapper is an extremely important part of how
    """
class test_broadcast_pytree(unittest.TestCase):
    """
    broadcast_pytree has complicated behavior with multiple
    helper functions. It needs its own test suite
    """
    @staticmethod
    def execute_replicate_adapter(source_tree: PyTree,
                                  target_tree: PyTree,
                                  context_msg: str
                                  ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        # A small adapter that takes forming the treedef into account
        source_leaves, source_treedef = jax.tree_util.tree_flatten(source_tree)
        target_treedef = jax.tree_util.tree_structure(target_tree)
        result, remaining_leaves = utils._replicate_leaves(source_treedef,
                                                           target_treedef,
                                                           source_leaves,
                                                           context_msg)
        return result, remaining_leaves
    def test_repeat_node(self):
        """Test that repeat node does it's namesake"""
        item = 1
        first, second, third = utils._repeat_node(item, 3)

        #Yes, should be is. We want the references to be the same
        self.assertTrue(item is first)
        self.assertTrue(item is second)
        self.assertTrue(item is third)
    def test_count_linked_leaves(self):
        """Test that count linked leaves detects the proper number of leaves on a pytree"""

        # Test edge case: No leaf
        tree = []
        _, tree_def = jax.tree_util.tree_flatten(tree)
        count = utils._count_linked_leaves(tree_def)
        self.assertTrue(count == 0)

        # Test case: One leaf
        tree = [3]
        _, tree_def = jax.tree_util.tree_flatten(tree)
        count = utils._count_linked_leaves(tree_def)
        self.assertTrue(count == 1)

        # Test case: Three leaves
        tree = [3, {"1" : 1, "2" : 2}, []]
        _, tree_def = jax.tree_util.tree_flatten(tree)
        count = utils._count_linked_leaves(tree_def)
        self.assertTrue(count == 3)
    def test_replicate_leaves(self):
        """ Perform a few tests on replicate leaves"""
        context = "An exception occurred while testing replicate leaves"

        # Edge case: Both start as tensor
        source = jnp.ones([3])
        target = jnp.ones([3, 7])
        expected = [source]
        outcome, remainder = self.execute_replicate_adapter(source,
                                                            target,
                                                            context)
        self.assertEqual(expected, outcome)
        self.assertEqual(remainder, [])

        #structure broadcast required to match structure

        source = jnp.ones([3])
        target = {"item" : [jnp.ones([3, 6]), jnp.ones([3,7])],
                  "item2" : jnp.ones([3,4])}
        expected = [source, source, source]

        outcome, remainder = self.execute_replicate_adapter(source,
                                                            target,
                                                            context)
        self.assertEqual(outcome, expected)
        self.assertEqual(remainder, [])

        # Broadcast from two locations

        source = {"branch1" : jnp.ones([3]), "branch2" : jnp.ones([5])}
        target = {"branch1" : [jnp.ones([3, 4]), jnp.ones([3, 5])],
                  "branch2" : [jnp.ones([3, 4])]}

        # Duplicate twice for branch one entry, once for branch two entry
        expected = [source["branch1"], source["branch1"], source["branch2"]]
        outcome, remainder = self.execute_replicate_adapter(source,
                                                            target,
                                                            context)
        self.assertEqual(outcome, expected)
        self.assertEqual(remainder, [])

        # Verify we function raise when working on states

        state = make_empty_state_mockup()
        tree = [state, state]
        output, remainder = self.execute_replicate_adapter(tree, tree, context)

        # Verify jit works

        execute_jit = jax.jit(self.execute_replicate_adapter, static_argnums=[2])
        output, remainder = execute_jit(tree, tree, context)
    def test_replicate_raises(self):
        """ Test that replicate leaves raises in various circumstances"""

        context_message = "Issue raised while testing replicate raises"

        # Raise due to contradictory keys

        source = {"item1" : 1, "item2" : 2}
        target = {"item2" : 1, "item3" : [3,5]}

        with self.assertRaises(RuntimeError) as err:
            self.execute_replicate_adapter(source, target, context_message)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)

        # Raise due to unequal lengths

        source = [1, 2, 3]
        target = [1, 2]
        with self.assertRaises(RuntimeError) as err:
            self.execute_replicate_adapter(source, target, context_message)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)

        # Raise due to different types
        source = [1, 2, 3]
        target = {"1" : 1, "2" : 2, "3" : 3}
        with self.assertRaises(RuntimeError) as err:
            self.execute_replicate_adapter(source, target, context_message)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)
    def test_broadcast_setup_pytree(self):
        """ Test that broadcast setup works reasonably well"""
        # Most of the complex tree logic is
        # dealt with in replicate leaves.

        # We will mainly be testing that the leaves are broadcast
        # right


        # Test efficiency edge case: scalar left broadcast, scalar right broadcast

        source_tree = jnp.array(2.0)
        target_tree = {"item1" : [jnp.ones([3, 2]), jnp.ones([3, 5])],
                "item2" : jnp.ones([3, 7])}

        left_outcome = utils.broadcast_pytree_shape(source_tree, target_tree)
        right_outcome = utils.broadcast_pytree_shape(source_tree, target_tree)

        self.assertEqual(left_outcome["item2"], source_tree)
        self.assertEqual(right_outcome["item2"], source_tree)

        # Test efficiency edge case: Tensor with ones for dimensions
        source_tree = jnp.ones([1, 1])

        target_tree = {"item1" : [jnp.ones([3, 2]), jnp.ones([3, 5])],
                "item2" : jnp.ones([3, 7])}

        left_outcome = utils.broadcast_pytree_shape(source_tree, target_tree)
        right_outcome = utils.broadcast_pytree_shape(source_tree, target_tree)

        self.assertEqual(left_outcome["item2"], source_tree)
        self.assertEqual(right_outcome["item2"], source_tree)

        # Test broadcast

        source_tree = {"item1" : jnp.ones([3]),
                       "item2" : jnp.ones([5])}
        target_tree = {"item1" : jnp.ones([3, 2, 5]),
                       "item2" : [jnp.ones([5, 2, 4]),
                                  jnp.ones([5, 4])]}
        expected_outcome = {"item1" : source_tree["item1"],
                            "item2" : [source_tree["item2"],
                                       source_tree["item2"]]
                            }

        left_outcome = utils.broadcast_pytree_shape(source_tree, target_tree)
        self.assertTrue(jnp.all(expected_outcome["item1"] == left_outcome["item1"]))
        self.assertTrue(jnp.all(expected_outcome["item2"][0] == left_outcome["item2"][0]))
        self.assertTrue(jnp.all(expected_outcome["item2"][1] == left_outcome["item2"][1]))

        # Test that jit works
        source_tree = jnp.ones([4])
        target_tree = jnp.ones([4, 5, 6])
        jit_function = jax.jit(utils.broadcast_pytree_shape)
        outcome = jit_function(source_tree, target_tree)
