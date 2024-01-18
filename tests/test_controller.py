"""
Unit tests to test the controller is functioning adequetely
"""

import unittest
import jax
from numpy import random
from jax import numpy as jnp
from src.states import ACTStates
from src.controller import ACT_Controller


def make_empty_state_mockup() -> ACTStates:
    return ACTStates(
        is_locked=None,
        epsilon=0,
        iterations=None,
        accumulators=None,
        defaults=None,
        updates=None,
        probabilities=None,
        residuals=None,
    )
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
            residuals = None,
            )
    def test_lock_indicator(self):
        """ Test if the lock indicator is correctly reflected """
        lock_status = True

        mock_state = make_empty_state_mockup()
        mock_state = mock_state.replace(is_locked=lock_status)

        controller = ACT_Controller(mock_state)
        self.assertEqual(lock_status, controller.is_locked)

    def test_residuals(self):
        """ Test if residuals are being read properly"""
        mock_residuals = jnp.array([0.1, 0.3, 0.4, 0.7])

        mock_state = make_empty_state_mockup()
        mock_state = mock_state.replace(residuals=mock_residuals)

        controller = ACT_Controller(mock_state)
        self.assertTrue(jnp.all(mock_residuals == controller.residuals))

    def test_probabilities(self):
        """ Test if probabilities are being read properly"""

        mock_probabilities = jnp.array([0.1, 0.3, 0.4, 0.7])

        mock_state = make_empty_state_mockup()
        mock_state = mock_state.replace(probabilities=mock_probabilities)

        controller = ACT_Controller(mock_state)
        self.assertTrue(jnp.all(mock_probabilities == controller.probabilities))

    def test_iterations(self):
        """ Test if the iteration state is being read properly"""

        mock_iterations = jnp.array([1, 10, 34])

        mock_state = make_empty_state_mockup()
        mock_state = mock_state.replace(iterations=mock_iterations)

        controller = ACT_Controller(mock_state)
        self.assertTrue(jnp.all(mock_iterations == controller.iterations))

    def test_accumulators(self):
        """ Test if the accumulator state is being read from properly"""

        mock_accumulator = {"matrix" : jnp.array([[0.1, 0.3],[0.2, 0.4]]),
                            "normalizer" : jnp.array([-0.3, 0.7])
                            }


        mock_state = make_empty_state_mockup()
        mock_state = mock_state.replace(accumulators=mock_accumulator)

        controller = ACT_Controller(mock_state)
        self.assertEqual(mock_accumulator, controller.accumulators)

    def test_halt_threshold(self):
        """ Test that the controller can sanely calculate a halt threshold from it's epsilon"""

        epsilon = 0.1
        expected_threshold = 1 - epsilon

        mock_state = make_empty_state_mockup()
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

        mock_state = make_empty_state_mockup()
        mock_state = mock_state.replace(epsilon = epsilon,
                                        probabilities=mock_probabilities)

        controller = ACT_Controller(mock_state)
        self.assertTrue(jnp.all(expected_halted == controller.halted_batches))


class test_private_helpers(unittest.TestCase):
    """
    Test that the private helper functions used by the
    controller are acting sanely.

    We mock up data to represent a condition, then see if the
    helpers act appropriately to produce a new state
    """
    def test_apply_mask(self):
        """ Test that apply mask will unsqueeze then apply a mask as is appropriate"""
        # The important thing here is that the mask should broadcast
        # from the first dimension in.

        mask = random.randn(2, 10) > 0
        data1 = random.randn(2, 10)
        data2 = random.randn(2, 10, 3, 14)

        ACT_Controller._apply_mask(mask, data1)
        ACT_Controller._apply_mask(mask, data2)

    def test_process_probabilities(self):
        """ Test that probability processing is acting as expected"""

        probabilities = jnp.array([0.4, 0.2, 0.7])
        halting_probs = jnp.array([0.3, 0.5, 0.8])
        residuals = jnp.array([0.0, 0.0, 0.0])

        # This has an epsilon that will only force
        # 0.7 + 0.8 into residual mode.

        epsilon = 0.1
        state = make_empty_state_mockup()
        state = state.replace(epsilon=epsilon,
                              probabilities=probabilities,
                              residuals=residuals)

        expected_probabilities = jnp.array([0.7, 0.7, 1.0])
        expected_halting_probabilities = jnp.array([0.3, 0.5, 0.3])
        expected_residuals = jnp.array([0.0, 0.0, 0.3])

        controller = ACT_Controller(state)
        halting_probs, residuals, probabilities = controller._process_probabilities(
                                                                halting_probs
                                                                )
        print(halting_probs)
        self.assertTrue(jnp.all(halting_probs==expected_halting_probabilities))
        self.assertTrue(jnp.all(probabilities==expected_probabilities))
        self.assertTrue(jnp.all(residuals == expected_residuals))


        halting_probs = 0.45

