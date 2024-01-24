"""
Unit tests to test the controller is functioning adequetely
"""

import jax
import unittest
from numpy import random
from jax import numpy as jnp
from jax.experimental import checkify

from src.jax_act import utils
from src.jax_act.states import ACTStates
from src.jax_act.controller import ACT_Controller

SHOW_ERROR_MESSAGES = True
def make_empty_state_mockup() -> ACTStates:
    return ACTStates(
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
            epsilon = 0,
            iterations=None,
            accumulators=None,
            defaults=None,
            updates=None,
            probabilities = None,
            residuals = None,
            )
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
    def test_validate_probabilities(self):
        """ Test that probabilities are validated as expected"""

        # Test we raise when too high
        with self.assertRaises(jax._src.checkify.JaxRuntimeError):
            err, _ = ACT_Controller.validate_probability(jnp.array([0, 3.0]))
            err.throw()

        # Test we raise when too low
        with self.assertRaises(jax._src.checkify.JaxRuntimeError):
            err, _ = ACT_Controller.validate_probability(jnp.array([0, -1.0]))
            err.throw()

        # Test we do not raise under valid conditions
        err, _ = ACT_Controller.validate_probability(jnp.array([0, 0.3]))
        err.throw()
    def test_process_probabilities(self):
        """ Test that probability processing is acting as expected"""

        probabilities = jnp.array([0.4, 0.2, 0.7])
        halting_probs = jnp.array([0.3, 0.5, 0.8])
        residuals = jnp.array([0.0, 0.0, 0.0])
        epsilon = 0.1

        # This has an epsilon that will only force
        # 0.7 + 0.8 into residual mode.
        #
        # We have manually proceeded this example, and
        # we see if it matches.

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

        self.assertTrue(jnp.allclose(halting_probs, expected_halting_probabilities))
        self.assertTrue(jnp.allclose(probabilities, expected_probabilities))
        self.assertTrue(jnp.allclose(residuals, expected_residuals))
    def test_update_accumulators(self):
        """ Test that update accumulators works in simple and pytree cases"""



        # Test validation logic: Detect when update was not ready to go

        epsilon = 0.1
        probabilities = jnp.array([0.1, 1.0])
        accumulator = jnp.array([[3.0, 4.0, -1.0],[0.7, 0.3, 0.5]])
        update = None
        halting_probabilities = jnp.array([0.6, 0.3])

        with self.assertRaises(RuntimeError) as err:
            state = make_empty_state_mockup()
            state = state.replace(epsilon = epsilon,
                                  probabilities=probabilities)

            mocked_controller = ACT_Controller(state)
            mocked_controller._update_accumulator(accumulator,
                                                  update,
                                                  halting_probabilities)
        if SHOW_ERROR_MESSAGES:
            print("Exception from test_update_accumulator: No cached update")
            print(err.exception)

        # Test differing shapes error condition


        epsilon = 0.1
        probabilities = jnp.array([0.1, 1.0])
        accumulator = jnp.array([[3.0, 4.0, -1.0], [0.7, 0.3, 0.5]])
        update = jnp.array([0.0, 1.0, 2.0])
        halting_probabilities = jnp.array([0.6, 0.3])

        with self.assertRaises(RuntimeError) as err:
            state = make_empty_state_mockup()
            state = state.replace(epsilon=epsilon,
                                  probabilities=probabilities)

            mocked_controller = ACT_Controller(state)
            mocked_controller._update_accumulator(accumulator,
                                                  update,
                                                  halting_probabilities)
        if SHOW_ERROR_MESSAGES:
            print("Exception from test_update_accumulator: Update shape different")
            print(err.exception)

        # Test that an actual update follows the correct logic
        #
        # It should multiply the update by the halting probabilities and
        # add. However, this should only occur when the halting probabilities
        # have not reached the exhausted state.

        epsilon = 0.1
        probabilities = jnp.array([0.1, 1.0])
        accumulator = jnp.array([[3.0, 4.0, -1.0], [0.7, 0.3, 0.5]])
        update = jnp.array([[2.0, 1.0, 0.1],[0.2, 10.0, 11.0]])
        halting_probabilities = jnp.array([0.5, 0.3])
        expected_output = jnp.array([[4.0, 4.5, -0.95],[0.7,0.3,0.5]])

        state = make_empty_state_mockup()
        state = state.replace(epsilon=epsilon,
                              probabilities=probabilities)

        mocked_controller = ACT_Controller(state)
        outcome = mocked_controller._update_accumulator(accumulator,
                                                        update,
                                                        halting_probabilities
                                                        )

        self.assertTrue(jnp.allclose(outcome, expected_output))

class test_main_logic(unittest.TestCase):
    """
    Test the main pieces of ACT logic that are used to perform
    act computation individually.
    """
    def test_cache_update(self):
        """ Test that updates are properly cached, and error handling goes off"""

        # Test that an error is thrown, as is proper, when
        # attempting to cache an update that was never configured

        mock_state = make_empty_state_mockup()
        mock_state = mock_state.replace(updates={})

        with self.assertRaises(ValueError) as err:
            update = jnp.array([0.0, 0.0, 0.0])

            controller = ACT_Controller(mock_state)
            controller.cache_update("Test", update)

        if SHOW_ERROR_MESSAGES:
            print("cache update test: No configured channel")
            print(err.exception)

        # Test that we correctly throw an error if the update was already set this
        # act iteration

        mock_state = make_empty_state_mockup()
        mock_state = mock_state.replace(updates={"test" : jnp.array([1.0, 2.0, 3.0])})

        with self.assertRaises(RuntimeError) as err:
            update = jnp.array([0.0, 0.0, 0.1])

            controller = ACT_Controller(mock_state)
            controller.cache_update("test", update)

        if SHOW_ERROR_MESSAGES:
            print("cache update test: Already assigned this iteration.")

        # Test that when these conditions do not occur, we correctly assign
        # the update to the given update channel
        update = jnp.array([0.0, 0.0, 0.1])

        mock_state = make_empty_state_mockup()
        mock_state = mock_state.replace(updates={"test" : None, "test2" : jnp.array([1.0, 2.0, 3.0])})

        controller = ACT_Controller(mock_state)
        controller = controller.cache_update("test", update)
        self.assertTrue(jnp.all(controller.state.updates["test"] == update))
    def test_act_iterate_error_handling(self):
        """ Test that the primary iteration method handles errors sanely"""

        # We start by setting up a sane state. We will corrupt it in various ways


        epsilon = 0.1
        iterations = jnp.array([3, 3, 2])
        probabilities = jnp.array([0.1, 0.5, 1.0])
        residuals = jnp.array([0.0,  0.0, 0.3])
        accumulator = {"state": jnp.array([[3.0, 4.0, -1.0], [0.5, 1.2, 2.1],[0.7, 0.3, 0.5]]),
                       "output" : jnp.array([[0.1, 0.1, 0.1], [1.0, 0.7, 0.3], [0.1, 0.1, 0.1]])
                       }
        update = {"state" : jnp.array([[1.0, 1.0, 1.0],[2.0, 2.0, 2.0], [-1.0, 2.0, 1.0]]),
                  "output" : jnp.array([[0.1, 0.2, 0.2],[1.2, 1.2, 1.2], [0.1, 0.1, 0.1]])}

        sane_state = make_empty_state_mockup()
        sane_state = sane_state.replace(epsilon = epsilon,
                                        iterations = iterations,
                                        residuals=residuals,
                                        probabilities=probabilities,
                                        accumulators=accumulator,
                                        updates=update)

        # Test that we detect halting probabilities are the wrong shape

        controller = ACT_Controller(sane_state)
        with self.assertRaises(ValueError) as err:
            halting_probabilities = random.randn(2, 3, 5)
            controller.iterate_act(halting_probabilities)

        if SHOW_ERROR_MESSAGES:
            print("iterate_act error message: When halting probabilities have wrong shape")
            print(err.exception)

        # Test we do NOT throw when halting probabilities are at 0.0 or 1.0

        controller = ACT_Controller(sane_state)
        halting_probabilities = jnp.array([0.0, 1.0, 0.5])
        controller.iterate_act(halting_probabilities)

        # Test we detect when an update was not cached

        missing_updates = sane_state.updates.copy()
        missing_updates["output"] = None
        no_updates_state = sane_state.replace(updates=missing_updates)

        controller = ACT_Controller(no_updates_state)

        with self.assertRaises(RuntimeError) as err:
            halting_probabilities = jnp.array([0.2, 0.3, 0.5])
            controller.iterate_act(halting_probabilities)


        if SHOW_ERROR_MESSAGES:
            print("iterate_act error message: Updates not fully prepped")
            print("Note: First message is top level, second is what traceback points to")
            print(err.exception)
            print(err.exception.__cause__)

        # Test we detect when a cached update is the wrong shape

        bad_updates = update.copy()
        bad_updates["output"] = random.randn(5, 10, 30)
        no_updates_state = sane_state.replace(updates=missing_updates)

        controller = ACT_Controller(no_updates_state)
        with self.assertRaises(RuntimeError) as err:
            halting_probabilities = jnp.array([0.2,0.3, 0.2])
            controller.iterate_act(halting_probabilities)

        if SHOW_ERROR_MESSAGES:
            print("iterate_act error message: Update has wrong shape")
            print("Note: First message is top level, second is what traceback points to")
            print(err.exception)
            print(err.exception.__cause__)
    def test_act_iterate(self):
        """ Test that the primary act iteration mechanism operates properly"""

        # We start by setting up all the various features for the state. We are
        # configuring a hypothetical ACT situation with a "state" and "output"
        # accumulators, which has two batch dimensions, and for which all updates
        # have already been setup.
        #
        # Additionally, the first batch will not enter the halted condition, the
        # second one is just now halting, and the third one is already halted.


        epsilon = 0.1
        iterations = jnp.array([3, 3, 2])
        probabilities = jnp.array([0.1, 0.5, 1.0])
        residuals = jnp.array([0.0,  0.0, 0.3])
        accumulator = {"state": jnp.array([[3.0, 4.0, -1.0], [0.5, 1.2, 2.1],[0.7, 0.3, 0.5]]),
                       "output" : jnp.array([[0.1, 0.1, 0.1], [1.0, 0.7, 0.3], [0.1, 0.1, 0.1]])
                       }
        update = {"state" : jnp.array([[1.0, 1.0, 1.0],[2.0, 2.0, 2.0], [-1.0, 2.0, 1.0]]),
                  "output" : jnp.array([[0.1, 0.2, 0.2],[1.2, 1.2, 1.2], [0.1, 0.1, 0.1]])}

        mock_state = make_empty_state_mockup()
        mock_state = mock_state.replace(epsilon = epsilon,
                                        iterations = iterations,
                                        residuals=residuals,
                                        probabilities=probabilities,
                                        accumulators=accumulator,
                                        updates=update)

        # The first halting probability should not be clamped. The second should be clamped to
        # 0.5, and the third should be clamped to zero.

        halting_probabilities = jnp.array([0.5, 0.7, 0.6])

        # The halting probability is designed to commit half of the update, where possible. That affects the
        # expected accumulator. We also expect to update probabilities


        expected_iterations = jnp.array([4, 4, 2])

        expected_accumulator = {"state" : jnp.array([[3.5, 4.5, -0.5],[1.5, 2.2, 3.1], [0.7, 0.3, 0.5]]),
                                "output" : jnp.array([[0.15, 0.2, 0.2], [1.6, 1.3, 0.9], [0.1, 0.1, 0.1]])
                                }
        expected_updates = {"state" : None, "output" : None}

        expected_residuals = jnp.array([0.0, 0.5, 0.3])
        expected_probabilities = jnp.array([0.6, 1.0, 1.0])

        # We actually need to run the tests

        controller = ACT_Controller(mock_state)
        new_controller = controller.iterate_act(halting_probabilities)

        print(new_controller)
        self.assertTrue(jnp.all(new_controller.iterations == expected_iterations))
        self.assertTrue(jnp.allclose(new_controller.probabilities, expected_probabilities))
        self.assertTrue(jnp.allclose(new_controller.residuals, expected_residuals))
        self.assertTrue(utils.are_pytrees_equal(new_controller.accumulators, expected_accumulator))
        self.assertTrue(utils.are_pytrees_equal(new_controller.state.updates, expected_updates))

    def test_reset_batches(self):
        """ Test that the reset batches method will isolate and reset to defaults appropriately."""

         # We start by setting up all the various features for the state. We are
        # configuring a hypothetical ACT situation with a "state" and "output"
        # accumulators, which has two batch dimensions, and for which all updates
        # have already been setup.
        #
        # Additionally, the first and second batch are not halted, but the third batch is
        #
        # We also have to define the defaults


        epsilon = 0.1
        iterations = jnp.array([3, 3, 2])
        probabilities = jnp.array([0.1, 0.4, 1.0])
        residuals = jnp.array([0.0,  0.0, 0.3])
        accumulator = {"state": jnp.array([[3.0, 4.0, -1.0], [0.5, 1.2, 2.1],[0.7, 0.3, 0.5]]),
                       "output" : jnp.array([[0.1, 0.1, 0.1], [1.0, 0.7, 0.3], [0.1, 0.1, 0.1]])
                       }
        defaults = {"state" : jnp.array([[0.0, 0.0, 0.0],[0.0,0.0,0.0], [0.0, 0.0, 0.0]]),
                    "output" : jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])}
        update = {"state" : None,
                  "output" : None}

        mock_state = make_empty_state_mockup()
        mock_state = mock_state.replace(epsilon = epsilon,
                                        iterations = iterations,
                                        residuals=residuals,
                                        probabilities=probabilities,
                                        accumulators=accumulator,
                                        defaults = defaults,
                                        updates=update)

        # Create expectations:
        #
        # Everything generally stays the same unless in the halted state.

        expected_iterations = jnp.array([3, 3, 0])
        expected_probabilities = jnp.array([0.1, 0.4, 0.0])
        expected_residuals = jnp.array([0.0, 0.0, 0.0])
        expected_accumulators = {"state": jnp.array([[3.0, 4.0, -1.0], [0.5, 1.2, 2.1],[0.0, 0.0, 0.0]]),
                       "output" : jnp.array([[0.1, 0.1, 0.1], [1.0, 0.7, 0.3], [1.0, 1.0, 1.0]])
                       }

        # Run test

        controller = ACT_Controller(mock_state)
        new_controller = controller.reset_batches()

        self.assertTrue(jnp.all(expected_iterations == new_controller.iterations))
        self.assertTrue(jnp.all(expected_probabilities == new_controller.probabilities))
        self.assertTrue(jnp.all(expected_residuals == new_controller.residuals))
        self.assertTrue(utils.are_pytrees_equal(expected_accumulators, new_controller.accumulators))

class test_jit(unittest.TestCase):
    """
    Test that the controller object can be integrated as a jitted object
    """
    def make_mock_state(self)->ACTStates:
        act_state = ACTStates(0.001,
                              jnp.array([0, 0]),
                              jnp.array([0.0, 0.0]),
                              jnp.array([0.0, 0.0]),
                              {"item" : jnp.array([0.0, 0.2])},
                              {"item" : jnp.array([0.0, 0.3])},
                              {"item" : None}

        )
        return act_state

    def test_return_controller(self):
        """
        Test we can return a controller from a function, that has been
        jitted
        """

        # Test we can make, and jit, a function
        # that will return a controller
        jax.config.update("jax_traceback_filtering", "off")

        state = self.make_mock_state()
        def make_controller(state):
            return ACT_Controller(state)

        jitted_controller = jax.jit(make_controller)
        controller = jitted_controller(state)

        # Test if we can still me meaningfully use that function
        controller = controller.cache_update("item", jnp.array([4.0, 0.3]))
        new_controller = controller.iterate_act(jnp.array([0.2, 0.3]))
        self.assertTrue(jnp.any(new_controller.probabilities != controller.probabilities))
        print(new_controller.probabilities)
    def test_cache_act(self):
        """ Test that cache act performs correctly when the method has been jitted."""
        update = jnp.array([0.0, 0.0, 0.1])

        mock_state = make_empty_state_mockup()
        mock_state = mock_state.replace(updates={"test" : None, "test2" : jnp.array([1.0, 2.0, 3.0])})

        controller = ACT_Controller(mock_state)
        def jit_cache_test(update: jnp.ndarray)->ACT_Controller:
            #NOTE: Must assign function to statically push into
            # test. Jax jit does not support jitting methods
            # with strings without using staticargnums.
            return controller.cache_update("test", update)

        jit_cache_update = jax.jit(jit_cache_test)
        controller = jit_cache_update(update)
        self.assertTrue(jnp.all(controller.state.updates["test"] == update))
    def test_access_properties(self):
        """ Test we can access and use properties in the jitted state"""

        jax.config.update("jax_traceback_filtering", "off")

        state = self.make_mock_state()

        def access_properties(state):
            controller = ACT_Controller(state)

            output = {}
            output["probabilities"] = controller.probabilities
            output["residuals"] = controller.residuals
            output["accumulator"] = controller.accumulators
            output["halted_batches"] = controller.halted_batches
            output["is_any_halted"] = controller.is_any_halted
            output["is_completely_halted"] = controller.is_completely_halted
            output["halt_threshold"] = controller.halt_threshold
            return output


        jitted_access = jax.jit(access_properties)
        properties = jitted_access(state)

    def test_iterate_act_jit(self):
        """ Test that iterate act is jittable"""


        # We start by setting up all the various features for the state. We are
        # configuring a hypothetical ACT situation with a "state" and "output"
        # accumulators, which has two batch dimensions, and for which all updates
        # have already been setup.
        #
        # Additionally, the first batch will not enter the halted condition, the
        # second one is just now halting, and the third one is already halted.


        epsilon = 0.1
        iterations = jnp.array([3, 3, 2])
        probabilities = jnp.array([0.1, 0.5, 1.0])
        residuals = jnp.array([0.0,  0.0, 0.3])
        accumulator = {"state": jnp.array([[3.0, 4.0, -1.0], [0.5, 1.2, 2.1],[0.7, 0.3, 0.5]]),
                       "output" : jnp.array([[0.1, 0.1, 0.1], [1.0, 0.7, 0.3], [0.1, 0.1, 0.1]])
                       }
        update = {"state" : jnp.array([[1.0, 1.0, 1.0],[2.0, 2.0, 2.0], [-1.0, 2.0, 1.0]]),
                  "output" : jnp.array([[0.1, 0.2, 0.2],[1.2, 1.2, 1.2], [0.1, 0.1, 0.1]])}

        mock_state = make_empty_state_mockup()
        mock_state = mock_state.replace(epsilon = epsilon,
                                        iterations = iterations,
                                        residuals=residuals,
                                        probabilities=probabilities,
                                        accumulators=accumulator,
                                        updates=update)

        # The first halting probability should not be clamped. The second should be clamped to
        # 0.5, and the third should be clamped to zero.

        halting_probabilities = jnp.array([0.5, 0.7, 0.6])

        # The halting probability is designed to commit half of the update, where possible. That affects the
        # expected accumulator. We also expect to update probabilities


        expected_iterations = jnp.array([4, 4, 2])

        expected_accumulator = {"state" : jnp.array([[3.5, 4.5, -0.5],[1.5, 2.2, 3.1], [0.7, 0.3, 0.5]]),
                                "output" : jnp.array([[0.15, 0.2, 0.2], [1.6, 1.3, 0.9], [0.1, 0.1, 0.1]])
                                }
        expected_updates = {"state" : None, "output" : None}

        expected_residuals = jnp.array([0.0, 0.5, 0.3])
        expected_probabilities = jnp.array([0.6, 1.0, 1.0])

        # We actually need to run the tests

        controller = ACT_Controller(mock_state)
        iterate_act_jit = jax.jit(controller.iterate_act)
        new_controller = iterate_act_jit(halting_probabilities)

        print(new_controller)
        self.assertTrue(jnp.all(new_controller.iterations == expected_iterations))
        self.assertTrue(jnp.allclose(new_controller.probabilities, expected_probabilities))
        self.assertTrue(jnp.allclose(new_controller.residuals, expected_residuals))
        self.assertTrue(utils.are_pytrees_equal(new_controller.accumulators, expected_accumulator))
        self.assertTrue(utils.are_pytrees_equal(new_controller.state.updates, expected_updates))
    def test_reset_batches_jit(self):

         # We start by setting up all the various features for the state. We are
        # configuring a hypothetical ACT situation with a "state" and "output"
        # accumulators, which has two batch dimensions, and for which all updates
        # have already been setup.
        #
        # Additionally, the first and second batch are not halted, but the third batch is
        #
        # We also have to define the defaults


        epsilon = 0.1
        iterations = jnp.array([3, 3, 2])
        probabilities = jnp.array([0.1, 0.4, 1.0])
        residuals = jnp.array([0.0,  0.0, 0.3])
        accumulator = {"state": jnp.array([[3.0, 4.0, -1.0], [0.5, 1.2, 2.1],[0.7, 0.3, 0.5]]),
                       "output" : jnp.array([[0.1, 0.1, 0.1], [1.0, 0.7, 0.3], [0.1, 0.1, 0.1]])
                       }
        defaults = {"state" : jnp.array([[0.0, 0.0, 0.0],[0.0,0.0,0.0], [0.0, 0.0, 0.0]]),
                    "output" : jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])}
        update = {"state" : None,
                  "output" : None}

        mock_state = make_empty_state_mockup()
        mock_state = mock_state.replace(epsilon = epsilon,
                                        iterations = iterations,
                                        residuals=residuals,
                                        probabilities=probabilities,
                                        accumulators=accumulator,
                                        defaults = defaults,
                                        updates=update)

        # Create expectations:
        #
        # Everything generally stays the same unless in the halted state.

        expected_iterations = jnp.array([3, 3, 0])
        expected_probabilities = jnp.array([0.1, 0.4, 0.0])
        expected_residuals = jnp.array([0.0, 0.0, 0.0])
        expected_accumulators = {"state": jnp.array([[3.0, 4.0, -1.0], [0.5, 1.2, 2.1],[0.0, 0.0, 0.0]]),
                       "output" : jnp.array([[0.1, 0.1, 0.1], [1.0, 0.7, 0.3], [1.0, 1.0, 1.0]])
                       }

        # Run test

        controller = ACT_Controller(mock_state)
        jit_reset_batches = jax.jit(controller.reset_batches)
        new_controller = jit_reset_batches()

        self.assertTrue(jnp.all(expected_iterations == new_controller.iterations))
        self.assertTrue(jnp.all(expected_probabilities == new_controller.probabilities))
        self.assertTrue(jnp.all(expected_residuals == new_controller.residuals))
        self.assertTrue(utils.are_pytrees_equal(expected_accumulators, new_controller.accumulators))

