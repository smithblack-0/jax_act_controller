"""
Tests for the editor act mechanism.

The editor is generally designed to allow the editing
of the values of an act process.
"""

import itertools
import unittest
import warnings
from typing import Callable, List, Optional

import jax

from jax import numpy as jnp
from jax.experimental import checkify
from src.jax_act.tensoreditor import TensorEditor, ErrorModes
from src.jax_act.states import ACTStates
from src.jax_act import utils
from src.jax_act.controller import ACT_Controller
from src.jax_act.types import PyTree

SHOW_ERROR_MESSAGES = False
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



class test_setter_validation(unittest.TestCase):
    """ Test that the validation functions operate properly"""
    @staticmethod
    def execute_validation(function: Callable):
        function = checkify.checkify(function)
        err, _ = function()
        err.throw()
    def test_validate_set_shape(self):
        item1 = jnp.zeros([10, 5])
        item2 = jnp.ones([10, 5])
        item3 = jnp.zeros([7, 5])

        info_message =  "Seen while testing validate shape"
        test_function = TensorEditor._validate_same_shape
        # Test we do not throw when shapes are compatible. Test under

        def validation():
            test_function(item1, item2, info_message)
        self.execute_validation(validation)

        validation = jax.jit(validation)
        self.execute_validation(validation)

        # Test we throw when they are not the same
        def validation():
            test_function(item1, item3, info_message)

        with self.assertRaises(checkify.JaxRuntimeError) as err:
            self.execute_validation(validation)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)

        validation = jax.jit(validation)
        with self.assertRaises(checkify.JaxRuntimeError) :
            self.execute_validation(validation)

    def test_validate_dtype(self):
        """ Test that validate dtype functions correctly"""
        item1 = jnp.zeros([10, 5], dtype=jnp.float32)
        item2 = jnp.ones([10, 5], dtype=jnp.float16)

        info_msg = "While testing validate dtype"
        target_function = TensorEditor._validate_same_dtype

        # Test we do not throw with compatible dtypes
        def validation():
            target_function(item1, item1, info_msg)
        self.execute_validation(validation)

        validation = jax.jit(validation)
        self.execute_validation(validation)

        # Test we do throw when required

        def validation():
            target_function(item1, item2, info_msg)

        with self.assertRaises(checkify.JaxRuntimeError) as err:
            self.execute_validation(validation)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)

        validation = jax.jit(validation)
        with self.assertRaises(checkify.JaxRuntimeError):
            self.execute_validation(validation)


    def test_validate_probabilities(self):
        """ Test that probability validation is good"""

        info_msg = "While testing validate probabilities"
        target_function = TensorEditor._validate_probability

        # Test that no throw happens when good
        valid_probabilities = jnp.array([0.0, 1.0, 0.3])
        def validate():
            target_function(valid_probabilities, info_msg)
        self.execute_validation(validate)

        validate = jax.jit(validate)
        self.execute_validation(validate)

        # Test we do throw when probabilities are too low

        invalid_probabilities = jnp.array([-0.1, 0.3, 0.4])
        def validate():
            target_function(invalid_probabilities, info_msg)
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            self.execute_validation(validate)
        if SHOW_ERROR_MESSAGES:
            print("Testing validate_probabilities: When probability too low")
            print(err.exception)

        validate = jax.jit(validate)
        with self.assertRaises(checkify.JaxRuntimeError):
            self.execute_validation(validate)

        # Test we do throw when probabilities are too high

        invalid_probabilities = jnp.array([1.3, 0.3, 0.4])
        def validate():
            target_function(invalid_probabilities, info_msg)
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            self.execute_validation(validate)
        if SHOW_ERROR_MESSAGES:
            print("Testing validate_probabilities: When probability too low")
            print(err.exception)

        validate = jax.jit(validate)
        with self.assertRaises(checkify.JaxRuntimeError):
            self.execute_validation(validate)
    def test_validate_is_natural_numbers(self):
        """ Test that validate whole numbers functions properly"""

        info_message = "Message generated while testing validate_is_natural_numbers"
        target_method = TensorEditor._validate_is_natural_numbers

        # Test we do not throw when we should not

        valid_numbers = jnp.array([0, 1, 2, 3, 5])
        def validate():
            target_method(valid_numbers, info_message)
        self.execute_validation(validate)

        # Test we throw when invalid
        invalid_numbers = jnp.array([-1, 2, 4])
        def validate():
            target_method(invalid_numbers, info_message)
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            self.execute_validation(validate)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)
    def test_validate_accumulator_exists(self):
        """ Test that validate accumulator exists functions properly"""

        info_message = "Message produced while testing_validate_accumulator_exists:"

        state = make_empty_state_mockup()
        accumulator = {"potato" : jnp.array([0.1, 0.2, 0.3])}
        state = state.replace(accumulators=accumulator, defaults=accumulator)


        # Test works properly when accumulator does exist
        def validate():
            editor = TensorEditor.edit_save(state)
            editor._validate_accumulator_exists("potato", info_message)
        self.execute_validation(validate)

        validate = jax.jit(validate)
        self.execute_validation(validate)

        # Test throws properly when accumulator does not exist
        def validate():
            editor = TensorEditor.edit_save(state)
            editor._validate_accumulator_exists("tomato", info_message)
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            self.execute_validation(validate)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)

        validate = jax.jit(validate)
        with self.assertRaises(checkify.JaxRuntimeError):
            self.execute_validation(validate)
    def test_validate_pytree_structure(self):
        """ This checks if pytrees have the same tree structure"""

        item1 = {"item" : jnp.zeros([3, 5])}
        item2 = {"item" : None, "baby" : None}
        item3 = {"item" : None}

        info_msg = "While testing validate pytree structure"
        target_method = TensorEditor._validate_same_pytree_structure
        # Check no throw when compatible
        #
        # Also check jit compatibity
        def validate():
            target_method(item1, item3, info_msg)
        self.execute_validation(validate)

        validate = jax.jit(validate)
        self.execute_validation(validate)

        # Check throws when incompatible
        def validate():
            target_method(item1, item2, info_msg)

        with self.assertRaises(checkify.JaxRuntimeError) as err:
            self.execute_validation(validate)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)

    def test_validate_pytree_leaves(self):
        """ See if validate pytree leaves is throwing when appropriate"""

        tensor_base = jnp.zeros([3, 2, 5])
        tensor_bad_shape = jnp.zeros([3, 2, 7])
        tensor_bad_dtype = jnp.zeros([3, 2, 5], dtype=jnp.float16)

        tree_base = {"item1" : tensor_base, "item2" : tensor_base}
        tree_bad_shapes = {"item1" : tensor_bad_shape, "item2" : tensor_bad_shape}
        tree_bad_dtype = {"item1" : tensor_bad_dtype, "item2" : tensor_bad_dtype}

        info_msg = "Testing validate pytree leaves"
        target_method = TensorEditor._validate_pytree_leaves

        # Test we do not throw when compatible
        def validation():
            target_method(tree_base, tree_base, info_msg)
        self.execute_validation(validation)

        validation = jax.jit(validation)
        self.execute_validation(validation)

        # test we throw with a bad shape

        def validation():
            target_method(tree_base, tree_bad_shapes, info_msg)
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            self.execute_validation(validation)
        if SHOW_ERROR_MESSAGES:
            print("testing validate_pytree_leaves: When shape is bad")
            print(err.exception)

        # test we throw on bad dtype
        def validation():
            target_method(tree_base, tree_bad_dtype, info_msg)
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            self.execute_validation(validation)

        if SHOW_ERROR_MESSAGES:
            print("testing validate_pytree_leaves: When dtype is bad")
            print(err.exception)

class test_setters(unittest.TestCase):
    """ Test that the setter functionality is behaving properly"""
    @staticmethod
    def error_modes()->List[str]:
        return [item.value for item in ErrorModes]
    @staticmethod
    def execute_validation(function: Callable):
        function = checkify.checkify(function)
        err, _ = function()
        err.throw()
    def make_state_mockup(self):
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
                                        defaults=accumulator,
                                        updates=update)
        return mock_state
    def test_set_probabilities(self):
        """ Test set_probabilities method in ControllerBuilder """

        # Initialize an editor in standard mode
        save = self.make_state_mockup()
        editor = TensorEditor.edit_save(save, "standard")

        # Set new values for probabilities

        new_probabilities = jnp.ones([3])
        updated_editor = editor.set_probabilities(new_probabilities)
        self.assertTrue(jnp.all(updated_editor.probabilities == new_probabilities))

        def validate():
            return editor.set_probabilities(new_probabilities)

        validate = jax.jit(validate)
        validate = checkify.checkify(validate)
        err, _ = validate()
        err.throw()

        # Attempt to set probabilities with values above one (should raise ValueError)
        above_one_probabilities = jnp.full([3], 1.1)  # Values greater than 1
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_probabilities(above_one_probabilities)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_probabilities: Values above one")
            print(err.exception)

        # Attempt to set probabilities with mismatched shape (should raise ValueError)
        mismatched_shape_probabilities = jnp.ones([4, 4])
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_probabilities(mismatched_shape_probabilities)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_probabilities: Mismatched shape")
            print(err.exception)

        # Attempt to set probabilities with mismatched dtype (should raise TypeError)
        mismatched_dtype_probabilities = jnp.ones([3, 3], dtype=jnp.int32)
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_probabilities(mismatched_dtype_probabilities)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_probabilities: Mismatched dtype")
            print(err.exception)


    def test_set_residuals(self):
        """ Test set_residuals method in ControllerBuilder """

        # Initialize a editor
        save = self.make_state_mockup()
        editor = TensorEditor.edit_save(save, "standard")


        # Set new values for residuals
        new_residuals = jnp.ones([3])
        updated_editor = editor.set_residuals(new_residuals)
        self.assertTrue(jnp.all(updated_editor.residuals == new_residuals))

        # Attempt to set residuals with mismatched shape (should raise ValueError)
        mismatched_shape_residuals = jnp.ones([4, 4])
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_residuals(mismatched_shape_residuals)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_residuals: Mismatched shape")
            print(err.exception)

        # Attempt to set residuals with mismatched dtype (should raise TypeError)
        mismatched_dtype_residuals = jnp.ones([3], dtype=jnp.int32)
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_residuals(mismatched_dtype_residuals)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_residuals: Mismatched dtype")
            print(err.exception)

        # Attempt to set residuals with values above one (should raise ValueError)
        above_one_residuals = jnp.full([3], 1.1)
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_residuals(above_one_residuals)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_residuals: Values above one")
            print(err.exception)

        # Attempt to set residuals with values below zero (should raise ValueError)
        below_zero_residuals = jnp.full([3], -0.1)
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_residuals(below_zero_residuals)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_residuals: Values below zero")
            print(err.exception)
    def test_set_iterations(self):
        """ Test set_iterations method in ControllerBuilder """

        # Initialize a editor
        save = self.make_state_mockup()
        editor = TensorEditor.edit_save(save, "standard")

        # Set new values for iterations (should be int32)
        new_iterations = jnp.array([1, 2, 3], dtype=jnp.int32)
        updated_editor = editor.set_iterations(new_iterations)
        self.assertTrue(jnp.all(updated_editor.iterations == new_iterations))

        # Attempt to set iterations with mismatched shape (should raise ValueError)
        mismatched_shape_iterations = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.int32)
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_iterations(mismatched_shape_iterations)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_iterations: Mismatched shape")
            print(err.exception)

        # Attempt to set iterations with mismatched dtype (should raise TypeError)
        mismatched_dtype_iterations = jnp.array([1, 2, 3], dtype=jnp.float32)
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_iterations(mismatched_dtype_iterations)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_iterations: Mismatched dtype")
            print(err.exception)

        # Attempt to set iterations when not a natural number
        not_natural_numbers = jnp.array([-1, 2, 3], dtype=jnp.int32)
        with self.assertRaises(checkify.JaxRuntimeError):
            editor.set_iterations(not_natural_numbers)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_iterations: Not natural numbers")
            print(err.exception)

    def test_set_epsilon(self):
        """ Test set_epsilon method in ControllerBuilder """

        # Initialize a editor
        # Initialize a editor
        save = self.make_state_mockup()
        editor = TensorEditor.edit_save(save, "standard")

        # Set a valid epsilon value
        new_epsilon = 0.5
        updated_editor = editor.set_epsilon(new_epsilon)
        self.assertEqual(updated_editor.state.epsilon, new_epsilon)

        # Attempt to set epsilon with a value less than 0 (should raise ValueError)
        below_zero_epsilon = -0.1
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_epsilon(below_zero_epsilon)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_epsilon: Value below zero")
            print(err.exception)

        # Attempt to set epsilon with a value greater than 1 (should raise ValueError)
        above_one_epsilon = 1.1
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_epsilon(above_one_epsilon)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_epsilon: Value above one")
            print(err.exception)

    def test_set_accumulator(self):
        """ Test set_accumulator method in ControllerBuilder """

        # Initialize a editor and define an accumulator
        initial_accumulator = {"matrix": jnp.zeros([3, 3, 3]), "normalizer": jnp.zeros([3, 3])}
        state = self.make_state_mockup()
        state = state.replace(accumulators={"test_acc" : initial_accumulator},
                              defaults = {"test_acc" : initial_accumulator})
        editor = TensorEditor.edit_save(state, "standard")

        # Set a new value for the defined accumulator
        new_accumulator_value = {"matrix": jnp.ones([3, 3, 3]), "normalizer": jnp.ones([3, 3])}
        updated_editor = editor.set_accumulator("test_acc", new_accumulator_value)
        self.assertTrue(jnp.all(updated_editor.accumulators["test_acc"]["matrix"] == jnp.ones([3, 3, 3])))
        self.assertTrue(jnp.all(updated_editor.accumulators["test_acc"]["normalizer"] == jnp.ones([3, 3])))

        # Attempt to set a non-existing accumulator (should raise KeyError)
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_accumulator("non_existing_acc", new_accumulator_value)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_accumulator: Non-existing accumulator")
            print(err.exception)

        # Attempt to set an accumulator with incompatible pytree (should raise ValueError)
        incompatible_accumulator_value = {"matrix": jnp.ones([4, 4, 4]), "normalizer": jnp.ones([3, 3])}
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_accumulator("test_acc", incompatible_accumulator_value)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_accumulator: Incompatible accumulator value")
            print(err.exception)
            print(err.exception.__cause__)
    def test_set_defaults(self):
        """ Test set_defaults method in ControllerBuilder """

        # Initialize a editor and define an accumulator with defaults
        state = self.make_state_mockup()
        initial_default = {}
        initial_default["test_acc"] = {"matrix": jnp.zeros([3, 3, 3]), "normalizer": jnp.zeros([3, 3])}
        state = state.replace(defaults = initial_default, accumulators=initial_default)
        editor = TensorEditor.edit_save(state)

        # Set new default values for the defined accumulator
        new_default_value = {"matrix": jnp.ones([3, 3, 3]), "normalizer": jnp.ones([3, 3])}
        updated_editor = editor.set_defaults("test_acc", new_default_value)
        self.assertTrue(jnp.all(updated_editor.defaults["test_acc"]["matrix"] == jnp.ones([3, 3, 3])))
        self.assertTrue(jnp.all(updated_editor.defaults["test_acc"]["normalizer"] == jnp.ones([3, 3])))

        # Attempt to set defaults for a non-existing accumulator (should raise KeyError)
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_defaults("non_existing_acc", new_default_value)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_defaults: Non-existing accumulator")
            print(err.exception)

        # Attempt to set defaults with incompatible pytree (should raise ValueError)
        incompatible_default_value = {"matrix": jnp.ones([4, 4, 4]), "normalizer": jnp.ones([3, 3])}
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_defaults("test_acc", incompatible_default_value)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_defaults: Incompatible default value")
            print(err.exception)
    def test_set_updates(self):
        """ Test set_updates method in ControllerBuilder """

        # Initialize a editor and define an accumulator with defaults
        state = self.make_state_mockup()
        initial_default = {}
        initial_default["test_acc"] = {"matrix": jnp.zeros([3, 3, 3]), "normalizer": jnp.zeros([3, 3])}
        state = state.replace(defaults = initial_default, accumulators=initial_default, updates=initial_default)
        editor = TensorEditor.edit_save(state)

        # Set new update values for the defined accumulator
        new_update_value = {"matrix": jnp.ones([3, 3, 3]), "normalizer": jnp.ones([3, 3])}
        updated_builder = editor.set_updates("test_acc", new_update_value)
        self.assertTrue(jnp.all(updated_builder.updates["test_acc"]["matrix"] == jnp.ones([3, 3, 3])))
        self.assertTrue(jnp.all(updated_builder.updates["test_acc"]["normalizer"] == jnp.ones([3, 3])))

        # Clear existing updates
        new_update_value = None
        updated_builder = editor.set_updates("test_acc", new_update_value)
        self.assertTrue(updated_builder.updates["test_acc"] is None)

        # Attempt to set updates for a non-existing accumulator (should raise KeyError)
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_updates("non_existing_acc", new_update_value)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_updates: Non-existing accumulator")
            print(err.exception)

        # Attempt to set updates with incompatible pytree (should raise ValueError)
        incompatible_update_value = {"matrix": jnp.ones([4, 4, 4]), "normalizer": jnp.ones([3, 3])}
        with self.assertRaises(checkify.JaxRuntimeError) as err:
            editor.set_updates("test_acc", incompatible_update_value)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_updates: Incompatible update value")
            print(err.exception)
            print(err.exception.__cause__)

class test_edit_cases(unittest.TestCase):
    """
    Test the effectiveness and possible span of the
    three error mechanisms.
    """

    @staticmethod
    def execute_validation(function: Callable):
        function = checkify.checkify(function)
        err, _ = function()
        err.throw()
    def make_state_mockup(self):
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
                                        defaults=accumulator,
                                        updates=update)
        return mock_state

    def test_moderate_edit(self):
        def edit_items(mode: str,
                       save: ACTStates,
                       probabilities: jnp.ndarray,
                       accumulator: PyTree,
                       updates: Optional[PyTree])->ACTStates:
            editor = TensorEditor.edit_save(save, mode)
            editor = editor.set_probabilities(probabilities)
            editor = editor.set_accumulator("output", accumulator)
            editor = editor.set_updates("output", updates)
            return editor.save()

        save = self.make_state_mockup()
        new_probabilities = jnp.ones([3])
        new_accumulator = jnp.ones([3, 3])

        # Test we function correctly under standard mode
        edit_items("standard", save, new_probabilities, new_accumulator, new_accumulator)
        edit_items = jax.jit(edit_items, static_argnums=[0])
        edit_items = checkify.checkify(edit_items)
        err, output = edit_items("standard", save, new_probabilities, new_accumulator, new_accumulator)
        self.assertTrue(jnp.all(output.probabilities == new_probabilities))

