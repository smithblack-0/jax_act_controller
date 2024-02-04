import unittest

import jax.tree_util
from numpy import random

from src.jax_act.core import utils
from jax import numpy as jnp

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

class test_broadcast_pytree(unittest.TestCase):
    """
    broadcast_pytree has complicated behavior with multiple
    helper functions. It needs its own test suite
    """
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
        tree = [3, {"1" : 1, "2" : 2}]
        _, tree_def = jax.tree_util.tree_flatten(tree)
        count = utils._count_linked_leaves(tree_def)
        self.assertTrue(count == 3)
    def test_replicate_leaves(self):
        """ Test for replicate leaves"""

        see = utils._replicate_leaves()



