"""
Unit tests to test the controller is functioning adequetely
"""

import unittest
import jax
from jax import numpy as jnp
from src.states import ACTStates
from src.controller import ACT_Controller

class test_properties(unittest.TestCase):
    """
    This is a relatively boring and uninformative series of tests
    that just verifies the properties work without throwing anything
    and yield expected content.

    However, the main logic functions depend on these being correct. Any
    error here would cause cascading issues. So we test.
    """

    def make_empty_state_mockup(self)->ACTStates:

        return ACTStates(
            is_locked = None,
            epsilon = 0,
            iterations=None,
            accumulators=None,
            defaults=None,
            updates=None,
            probabilities = None,
            )
    def test_lock_indicator(self):
        """ Test if the lock indicator is correctly reflected """
        lock_status = True

        mock_state = self.make_empty_state_mockup()
        mock_state = mock_state.replace(is_locked=lock_status)

        controller = ACT_Controller(mock_state)
        self.assertEqual(lock_status, controller.is_locked)

    def test_residuals(self):
        """ Test if residuals are being read properly"""
        mock_residuals = jnp.array([0.1, 0.3, 0.4, 0.7])

        mock_state = self.make_empty_state_mockup()
        mock_state = mock_state.replace(residuals=mock_probabilities)

        controller = ACT_Controller(mock_state)
        self.assertEqual(mock_probabilities, controller.residuals)

    def test_probabilities(self):
        """ Test if probabilities are being read properly"""

        mock_probabilities = jnp.array([0.1, 0.3, 0.4, 0.7])

        mock_state = self.make_empty_state_mockup()
        mock_state = mock_state.replace(probabilities=mock_probabilities)

        controller = ACT_Controller(mock_state)
        self.assertEqual(mock_probabilities, controller.probabilities)

    def test_iterations(self):
        """ Test if the iteration state is being read properly"""

        mock_iterations = jnp.array([1, 10, 34])

        mock_state = self.make_empty_state_mockup()
        mock_state = mock_state.replace(iterations=mock_iterations)

        controller = ACT_Controller(mock_state)
        self.assertEqual(mock_iterations, controller.iterations)

    def test_accumulators(self):
        """ Test if the accumulator state is being read from properly"""

        mock_accumulator = {"matrix" : jnp.array([[0.1, 0.3],[0.2, 0.4]]),
                            "normalizer" : jnp.array([-0.3, 0.7])
                            }


        mock_state = self.make_empty_state_mockup()
        mock_state = mock_state.replace(accumulators=mock_accumulator)

        controller = ACT_Controller(mock_state)
        self.assertEqual(mock_accumulator, controller.accumulators)

    def test_halt_threshold(self):
        """ Test that the controller can sanely calculate a halt threshold from it's epsilon"""

        epsilon = 0.1
        expected_threshold = 1 - epsilon

        mock_state = self.make_empty_state_mockup()
        mock_state = mock_state.replace(epsilon = epsilon)

        controller = ACT_Controller(mock_state)

        self.assertEqual(expected_threshold, controller.halt_threshold)

    def test_halt_batches(self):
        """ Test that the controller can correctly identify which batches are operating in the halted
            condition."""

        # Batches with a cumulative probability above the halt threshold should have been marked
        # as halted.

        epsilon = 0.1
        halt_threshold = 1 - epsilon

        mock_probabilities = jnp.array([0.1, 0.3, 0.4, 0.99])
        expected_halted = jnp.array([False, False, False, True])

        mock_state = self.make_empty_state_mockup()
        mock_state = mock_state.replace(epsilon = epsilon,
                                        probabilities=mock_probabilities)

        controller = ACT_Controller(mock_state)

        self.assertEqual(expected_halted, controller.halted_batches)


class test_private_helpers(unittest.TestCase):
    """
    Test that the private helper functions used by the
    controller are acting sanely.

    We mock up data to represent a condition, then see if the
    helpers act appropriately.
    """

