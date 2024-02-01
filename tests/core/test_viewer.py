"""
Tests for seeing if the ACT Viewer is working
sanely.
"""
import unittest
import jax
import numpy as np

from src.jax_act.states import ACTStates
from src.jax_act.viewer import ACTViewer
from src.jax_act import utils
from jax import numpy as jnp

class test_properties(unittest.TestCase):
    def make_mock_state(self)->ACTStates:
        epsilon = 0.1
        iterations = jnp.array([3, 3, 2])
        probabilities = jnp.array([0.1, 0.5, 1.0])
        residuals = jnp.array([0.0,  0.0, 0.3])
        accumulator = {"state": jnp.array([[3.0, 4.0, -1.0], [0.5, 1.2, 2.1],[0.7, 0.3, 0.5]]),
                       "output" : jnp.array([[0.1, 0.1, 0.1], [1.0, 0.7, 0.3], [0.1, 0.1, 0.1]])
                       }
        defaults = {"state" : jnp.array([[1.0, 1.0, 1.0],[2.0, 2.0, 2.0], [-1.0, 2.0, 1.0]]),
                  "output" : jnp.array([[0.1, 0.2, 0.2],[1.2, 1.2, 1.2], [0.1, 0.1, 0.1]])}

        updates = {"state" : jnp.array([[1.0, 1.0, 1.0],[2.0, 2.0, 2.0], [-1.0, 2.0, 1.0]]),
                  "output" : None}

        state = ACTStates(epsilon=epsilon,
                          iterations=iterations,
                          probabilities=probabilities,
                          residuals=residuals,
                          accumulators=accumulator,
                          defaults=defaults,
                          updates=updates)
        return state



    def setUp(self):
        # Storing mock state for use in each test
        self.mock_state = self.make_mock_state()
        self.viewer = ACTViewer(self.mock_state)

    def test_probabilities_property(self):
        self.assertTrue(jnp.all(self.viewer.probabilities == self.mock_state.probabilities))

    def test_residuals_property(self):
        self.assertTrue(jnp.all(self.viewer.residuals == self.mock_state.residuals))

    def test_iterations_property(self):
        self.assertTrue(jnp.all(self.viewer.iterations == self.mock_state.iterations))

    def test_accumulators_property(self):
        self.assertTrue(utils.are_pytrees_equal(self.viewer.accumulators, self.mock_state.accumulators))

    def test_defaults_property(self):
        self.assertTrue(utils.are_pytrees_equal(self.viewer.defaults, self.mock_state.defaults))

    def test_updates_property(self):
        self.assertTrue(utils.are_pytrees_equal(self.viewer.updates, self.mock_state.updates))

    def test_halt_threshold_property(self):
        expected_halt_threshold = 1 - self.mock_state.epsilon
        self.assertEqual(self.viewer.halt_threshold, expected_halt_threshold)

    def test_halted_batches_property(self):
        expected_halted_batches = self.viewer.probabilities > (1 - self.mock_state.epsilon)
        self.assertTrue(jnp.all(self.viewer.halted_batches ==expected_halted_batches))

class test_viewer_methods(unittest.TestCase):
    def make_mock_state(self) -> ACTStates:
        epsilon = 0.1
        iterations = jnp.array([3, 3, 2])
        probabilities = jnp.array([0.1, 0.5, 1.0])
        residuals = jnp.array([0.0, 0.0, 0.3])
        accumulator = {"state": jnp.array([[3.0, 4.0, -1.0], [0.5, 1.2, 2.1], [0.7, 0.3, 0.5]]),
                       "output": jnp.array([[0.1, 0.1, 0.1], [1.0, 0.7, 0.3], [0.1, 0.1, 0.1]])
                       }
        defaults = {"state": jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [-1.0, 2.0, 1.0]]),
                    "output": jnp.array([[0.1, 0.2, 0.2], [1.2, 1.2, 1.2], [0.1, 0.1, 0.1]])}

        updates = {"state": jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [-1.0, 2.0, 1.0]]),
                   "output": None}

        state = ACTStates(epsilon=epsilon,
                          iterations=iterations,
                          probabilities=probabilities,
                          residuals=residuals,
                          accumulators=accumulator,
                          defaults=defaults,
                          updates=updates)
        return state
    def setUp(self):
        # Storing mock state for use in each test
        self.mock_state = self.make_mock_state()
        self.viewer = ACTViewer(self.mock_state)

    def test_transform_throws(self):
        """
        Test that transform throws when the provided function raises something
        """

        def faulty_function(tensor):
            raise Exception("Intentional Error")

        with self.assertRaises(RuntimeError) as context:
            self.viewer.transform(faulty_function)

        self.assertIn("An issue occurred in using the provided function", str(context.exception))

    def test_transform(self):
        """
        Test that transform is operational.
        """

        def reshape_function(tensor):
            return tensor[None, ...]

        transformed_viewer = self.viewer.transform(reshape_function)

        # Check if the reshaped tensors match the expected shape
        self.assertEqual(transformed_viewer.iterations.shape, (1, 3))
        self.assertEqual(transformed_viewer.residuals.shape, (1, 3))
        self.assertEqual(transformed_viewer.probabilities.shape, (1, 3))

        # Check for accumulators, defaults, and updates
        for _, value in transformed_viewer.accumulators.items():
            for v in jax.tree_util.tree_leaves(value):
                self.assertEqual(v.shape[0:2], (1, 3))

        for _, value in transformed_viewer.defaults.items():
            for v in jax.tree_util.tree_leaves(value):
                self.assertEqual(v.shape[0:2], (1, 3))

        for _, value in transformed_viewer.updates.items():
            for v in jax.tree_util.tree_leaves(value):
                if v is not None:
                    self.assertEqual(v.shape[0:2], (1, 3))

    def test_mask_throws(self):
        """
        Test that mask throws when appropriate
        """
        # Case 1: Mask with incorrect dtype
        incorrect_dtype_mask = jnp.array([1, 0, 1])  # Not a boolean array
        with self.assertRaises(ValueError) as context:
            self.viewer.mask_data(incorrect_dtype_mask)
        self.assertIn("Mask was not made up of bool dtypes", str(context.exception))

        # Case 2: Mask with incorrect shape
        incorrect_shape_mask = jnp.array([True, False])  # Shape different from batch shape
        with self.assertRaises(ValueError) as context:
            self.viewer.mask_data(incorrect_shape_mask)
        self.assertIn("does not match batch shape", str(context.exception))

    def test_mask(self):
        """
        Test that the mask function works
        """
        # Mask out element 1
        mask_1 = jnp.array([False, True, True])
        masked_viewer_1 = self.viewer.mask_data(mask_1)
        self.assertEqual(masked_viewer_1.probabilities[0], 0)

        # Other elements should remain unchanged
        self.assertEqual(masked_viewer_1.probabilities[1], self.viewer.probabilities[1])
        self.assertEqual(masked_viewer_1.probabilities[2], self.viewer.probabilities[2])

        # Mask all elements
        mask_all = jnp.array([False, False, False])
        masked_viewer_all = self.viewer.mask_data(mask_all)
        np.testing.assert_array_equal(masked_viewer_all.probabilities, jnp.array([0, 0, 0]))

        # Mask no elements
        mask_none = jnp.array([True, True, True])
        masked_viewer_none = self.viewer.mask_data(mask_none)
        np.testing.assert_array_equal(masked_viewer_none.probabilities, self.viewer.probabilities)

    def test_unhalted_only(self):
        """ Test that in progress only works"""
        # Minimal test is needed, as this is mostly a wrapper around already validated functions
        masked_viewer = self.viewer.unhalted_only()
        self.assertTrue(masked_viewer.probabilities[2] == 0)
        self.assertTrue(masked_viewer.probabilities[1] == 0.5)

    def test_halted_only(self):
        """Test that finished only works"""
        # Minimal test is needed, as this is mostly a wrapper around already validated functions
        masked_viewer = self.viewer.halted_only()
        self.assertTrue(masked_viewer.probabilities[2] == 1.0)
        self.assertTrue(masked_viewer.probabilities[1] == 0)
        self.viewer.halted_only()


class test_viewer_jit(unittest.TestCase):
    def make_mock_state(self) -> ACTStates:
        epsilon = 0.1
        iterations = jnp.array([3, 3, 2])
        probabilities = jnp.array([0.1, 0.5, 1.0])
        residuals = jnp.array([0.0, 0.0, 0.3])
        accumulator = {"state": jnp.array([[3.0, 4.0, -1.0], [0.5, 1.2, 2.1], [0.7, 0.3, 0.5]]),
                       "output": jnp.array([[0.1, 0.1, 0.1], [1.0, 0.7, 0.3], [0.1, 0.1, 0.1]])
                       }
        defaults = {"state": jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [-1.0, 2.0, 1.0]]),
                    "output": jnp.array([[0.1, 0.2, 0.2], [1.2, 1.2, 1.2], [0.1, 0.1, 0.1]])}

        updates = {"state": jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [-1.0, 2.0, 1.0]]),
                   "output": None}

        state = ACTStates(epsilon=epsilon,
                          iterations=iterations,
                          probabilities=probabilities,
                          residuals=residuals,
                          accumulators=accumulator,
                          defaults=defaults,
                          updates=updates)
        return state

    def setUp(self):
        # Storing mock state for use in each test
        self.mock_state = self.make_mock_state()
        self.viewer = ACTViewer(self.mock_state)

    def test_properties(self):
        """ Test that all the properties work"""

        # Define jit getters
        probability_getter = jax.jit(lambda : self.viewer.probabilities)
        residuals_getter = jax.jit(lambda : self.viewer.residuals)
        iterations_getter = jax.jit(lambda : self.viewer.iterations)
        accumulators_getter = jax.jit(lambda : self.viewer.accumulators)
        defaults_getter = jax.jit(lambda : self.viewer.defaults)
        updates_getter = jax.jit(lambda : self.viewer.updates)
        halt_threshold_getter = jax.jit(lambda : self.viewer.halt_threshold)
        halted_batches_getter = jax.jit(lambda : self.viewer.halted_batches)

        # Use them.
        probability_getter()
        residuals_getter()
        iterations_getter()
        accumulators_getter()
        defaults_getter()
        updates_getter()
        halted_batches_getter()
        halt_threshold_getter()

    def test_transform_jit(self):
        """
        Test if transform can operate correctly.
        """

        # Define the transform to perform

        @jax.tree_util.Partial
        def expand(tensor: jnp.ndarray)->jnp.ndarray:
            return tensor[None, ...]

        # Jit the method

        transform_jit = jax.jit(self.viewer.transform)

        # Apply it. Then verify the underlying data was updated
        new_viewer = transform_jit(expand)
        self.assertEqual(new_viewer.probabilities.shape[:2], (1, 3))

    def test_mask_data_jit(self):
        """Test that mask functions correctly"""

        # Define the mask

        mask = jnp.array([True, True, False])

        #Jit the method
        mask_data_jit = jax.jit(self.viewer.mask_data)

        # Apply, test
        new_viewer = mask_data_jit(mask)
        self.assertEqual(new_viewer.probabilities[1], 0.5)
        self.assertEqual(new_viewer.probabilities[2], 0.0)

    def test_halted_only_jit(self):
        """Test that halted only jits correctly"""

        # Jit the method
        halted_only_jit = jax.jit(self.viewer.halted_only)

        # Apply, test
        new_viewer = halted_only_jit()
        self.assertEqual(new_viewer.probabilities[1], 0.0)
        self.assertEqual(new_viewer.probabilities[2], 1.0)

    def test_unhalted_only_jit(self):
        """ Test that unhalted only jits correctly"""

        # Jit the method

        unhalted_only_jit = jax.jit(self.viewer.unhalted_only)

        # Apply, test
        new_viewer = unhalted_only_jit()
        self.assertEqual(new_viewer.probabilities[1], 0.5)
        self.assertEqual(new_viewer.probabilities[2], 0.0)
