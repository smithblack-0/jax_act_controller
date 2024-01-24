"""
Tests for the builder act mechanism
"""

import itertools
import unittest
import warnings
import jax

from jax import numpy as jnp
from src.jax_act.builder import ControllerBuilder
from src.jax_act.states import ACTStates
from src.jax_act import utils
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

class test_setter_validation(unittest.TestCase):
    def test_validate_set_shape(self):
        item1 = jnp.zeros([10, 5])
        item2 = jnp.ones([10, 5])
        item3 = jnp.zeros([7, 5])

        # Test we do not throw where shapes are compatible

        ControllerBuilder._validate_set_shape(item1, item2)

        # Test we do when they are not

        with self.assertRaises(ValueError) as err:
            ControllerBuilder._validate_set_shape(item1, item3)

        if SHOW_ERROR_MESSAGES:
            print(err.exception)

    def test_validate_dtype(self):
        item1 = jnp.zeros([10, 5], dtype=jnp.float32)
        item2 = jnp.ones([10, 5], dtype=jnp.float16)

        # Test we do not throw with compatible dtypes

        ControllerBuilder._validate_set_dtype(item1, item1)

        # Test we do throw when required

        with self.assertRaises(TypeError) as err:
            ControllerBuilder._validate_set_dtype(item1, item2)

        if SHOW_ERROR_MESSAGES:
            print(err.exception)

    def test_validate_pytree_structure(self):
        """ This checks if pytrees have the same tree structure"""

        item1 = {"item" : jnp.zeros([3, 5])}
        item2 = {"item" : None, "baby" : None}
        item3 = {"item" : None}

        # Check no throw when compatible
        ControllerBuilder._validate_same_pytree_structure(item1, item3)

        # Check throw when incompatible
        with self.assertRaises(ValueError) as err:
            ControllerBuilder._validate_same_pytree_structure(item1, item2)
        if SHOW_ERROR_MESSAGES:
            print(err.exception)

    def test_validate_pytree_leaves(self):
        """ See if validate pytree leaves is throwing when appropriate"""

        tensor_base = jnp.zeros([3, 2, 5])
        tensor_bad_shape = jnp.zeros([3, 2, 7])
        tensor_bad_dtype = jnp.zeros([3, 2, 5], dtype=jnp.float16)

        tree_base = {"item1" : tensor_base, "item2" : tensor_base}
        tree_bad_type = {"item1" : tensor_base, "item2" : "test"}
        tree_bad_shapes = {"item1" : tensor_bad_shape, "item2" : tensor_bad_shape}
        tree_bad_dtype = {"item1" : tensor_bad_dtype, "item2" : tensor_bad_dtype}

        # Test we do not throw when compatible
        ControllerBuilder._validate_pytree_leaves(tree_base, tree_base)

        # Test we throw with bad type
        with self.assertRaises(TypeError) as err:
            ControllerBuilder._validate_pytree_leaves(tree_base, tree_bad_type)

        if SHOW_ERROR_MESSAGES:
            print("testing validate_pytree_leaves: When type is bad")
            print(err.exception)

        # test we throw with a bad shape

        with self.assertRaises(ValueError) as err:
            ControllerBuilder._validate_pytree_leaves(tree_base, tree_bad_shapes)

        if SHOW_ERROR_MESSAGES:
            print("testing validate_pytree_leaves: When shape is bad")
            print(err.exception)

        # test we throw on bad dtype

        with self.assertRaises(TypeError) as err:
            ControllerBuilder._validate_pytree_leaves(tree_base, tree_bad_dtype)

        if SHOW_ERROR_MESSAGES:
            print("testing validate_pytree_leaves: When dtype is bad")
            print(err.exception)


    def test_validate_definition_pytree(self):
        """ Test if _validate_definition_pytree throws errors appropriately """

        batch_shape = [3, 2]
        good_tensor = jnp.zeros(batch_shape, dtype=jnp.float32)
        bad_type_tensor = "not a tensor"
        bad_dtype_tensor = jnp.zeros(batch_shape, dtype=jnp.int32)
        bad_shape_tensor = jnp.zeros((5, 3), dtype=jnp.float32)

        good_tree = {"leaf1": good_tensor, "leaf2": good_tensor}
        bad_type_tree = {"leaf1": good_tensor, "leaf2": bad_type_tensor}
        bad_dtype_tree = {"leaf1": good_tensor, "leaf2": bad_dtype_tensor}
        bad_shape_tree = {"leaf1": good_tensor, "leaf2": bad_shape_tensor}

        builder = ControllerBuilder.new_builder(batch_shape)  # Initialize with appropriate parameters

        # Test that valid pytree does not throw
        builder._validate_definition_pytree(good_tree)

        # Test for invalid tensor type
        with self.assertRaises(TypeError) as err:
            builder._validate_definition_pytree(bad_type_tree)
        if SHOW_ERROR_MESSAGES:
            print("Testing _validate_definition_pytree: Invalid Tensor Type")
            print(err.exception)

        # Test for invalid tensor dtype
        with self.assertRaises(TypeError) as err:
            builder._validate_definition_pytree(bad_dtype_tree)
        if SHOW_ERROR_MESSAGES:
            print("Testing _validate_definition_pytree: Invalid Tensor Dtype")
            print(err.exception)

        # Test for invalid tensor shape
        with self.assertRaises(ValueError) as err:
            builder._validate_definition_pytree(bad_shape_tree)
        if SHOW_ERROR_MESSAGES:
            print("Testing _validate_definition_pytree: Invalid Tensor Shape")
            print(err.exception)

class test_instance_methods(unittest.TestCase):
    """
    Test that methods which are suppose to return a new instance
    do in fact work.
    """
    def test_new_builder(self):
        """ Test that new builder works with a few various conditions"""

        dtypes = [jnp.float32, jnp.float16]
        epsilons = [0.1, 0.3, 0.7]
        batch_shapes = [10, [10, 20], [3]]

        for dtype, epsilon, batch_shape in itertools.product(dtypes, epsilons, batch_shapes):
            expected_tensor = jnp.zeros(batch_shape, dtype)
            expected_iteration = jnp.zeros(batch_shape, jnp.int32)

            builder = ControllerBuilder.new_builder(batch_shape, dtype, epsilon)

            self.assertTrue(builder.epsilon == epsilon)

            self.assertTrue(jnp.all(builder.probabilities == expected_tensor))
            self.assertTrue(jnp.all(builder.residuals == expected_tensor))
            self.assertTrue(jnp.all(builder.iterations == expected_iteration))

            self.assertTrue(isinstance(builder.accumulators, dict))
            self.assertTrue(isinstance(builder.defaults, dict))
            self.assertTrue(isinstance(builder.updates, dict))

            self.assertTrue(len(builder.accumulators) == 0)
            self.assertTrue(len(builder.updates) == 0)
            self.assertTrue(len(builder.defaults) == 0)

    def test_new_builder_validation(self):
        """ Test that new builder throws under its validation conditions """

        # Case 1: core_dtype is not a floating type
        with self.assertRaises(ValueError) as err:
            ControllerBuilder.new_builder(batch_shape=[3, 3], core_dtype=jnp.int32)
        if SHOW_ERROR_MESSAGES:
            print("Testing new_builder: core_dtype not floating")
            print(err.exception)

        # Case 2: epsilon is not between 0 and 1 (testing with a value greater than 1)
        with self.assertRaises(ValueError) as err:
            ControllerBuilder.new_builder(batch_shape=[3, 3], epsilon=1.5)
        if SHOW_ERROR_MESSAGES:
            print("Testing new_builder: epsilon greater than 1")
            print(err.exception)

        # Case 3: epsilon is not between 0 and 1 (testing with a value less than 0)
        with self.assertRaises(ValueError) as err:
            ControllerBuilder.new_builder(batch_shape=[3, 3], epsilon=-0.1)
        if SHOW_ERROR_MESSAGES:
            print("Testing new_builder: epsilon less than 0")
            print(err.exception)



class test_definition_methods(unittest.TestCase):
    """
    Tests for the methods that define or remove accumulators.
    """
    def test_define_private(self):
        """ Test that the private define function is acting sanely"""

        batch_shape = 10
        tensor = jnp.zeros([batch_shape, 3, 4])
        tensor_two = jnp.ones([batch_shape, 2, 4])


        # Test new definition
        definition = {"items1" : tensor, "items2" : tensor_two}

        builder = ControllerBuilder.new_builder(batch_shape)
        builder = builder._define("test_accumulator", definition)

        self.assertTrue(utils.are_pytrees_equal(builder.accumulators["test_accumulator"], definition))
        self.assertTrue(utils.are_pytrees_equal(builder.defaults["test_accumulator"], definition))
        self.assertTrue(builder.updates["test_accumulator"] is None)

        # Test if we overwrite the existing definition

        new_definition = {"items1" : tensor_two}
        builder = builder._define("test_accumulator", new_definition)

        self.assertTrue(utils.are_pytrees_equal(builder.accumulators["test_accumulator"], new_definition))
        self.assertTrue(utils.are_pytrees_equal(builder.defaults["test_accumulator"], new_definition))
        self.assertTrue(builder.updates["test_accumulator"] is None)

    def test_define_accumulator_directly(self):
        """ Test define_accumulator_directly method for various scenarios """

        # Initialize a builder for testing
        builder = ControllerBuilder.new_builder(batch_shape=[3, 3])

        # Define a valid accumulator
        valid_accumulator = {"matrix": jnp.zeros([3, 3, 3]), "normalizer": jnp.zeros([3, 3])}
        new_builder = builder.define_accumulator_directly("valid_acc", valid_accumulator)
        self.assertIn("valid_acc", new_builder.defaults)
        self.assertIn("valid_acc", new_builder.accumulators)

        # Overwrite an existing accumulator (should raise a warning)
        with warnings.catch_warnings(record=True) as w:
            new_builder = new_builder.define_accumulator_directly("valid_acc", valid_accumulator)
        if SHOW_ERROR_MESSAGES:
            for warn in w:
                print(f"Warning caught: {warn.message}")

        # Attempt to define an accumulator with an invalid definition (e.g., non-tensor type)
        invalid_accumulator = {"matrix": [1, 2, 3], "normalizer": jnp.zeros([3, 3])}
        with self.assertRaises(TypeError) as err:
            new_builder.define_accumulator_directly("invalid_acc", invalid_accumulator)
        if SHOW_ERROR_MESSAGES:
            print("Testing define_accumulator_directly: Invalid accumulator definition")
            print(err.exception)
            print(err.exception.__cause__)

    def test_define_accumulator_by_shape(self):
        """ Test define_accumulator_by_shape method for various scenarios """

        # Initialize a builder for testing
        builder = ControllerBuilder.new_builder(batch_shape=[3, 3])

        # Define a valid accumulator by shape
        valid_shape = [3, 3, 3]
        expected_accumulator = jnp.zeros(valid_shape)
        new_builder = builder.define_accumulator_by_shape("valid_acc", valid_shape)
        self.assertIn("valid_acc", new_builder.defaults)
        self.assertTrue(jnp.all(new_builder.accumulators["valid_acc"] == expected_accumulator))

        # Overwrite an existing accumulator (should raise a warning)
        with warnings.catch_warnings(record=True) as w:
            new_builder = new_builder.define_accumulator_by_shape("valid_acc", valid_shape)
        if SHOW_ERROR_MESSAGES:
            for warn in w:
                print(f"Warning caught: {warn.message}")

        # Define a valid accumulator, but this time use a pytree to do it
        valid_pytree_shapes = {"items1" : [3, 3, 4], "item2" : [3, 3, 10, 4]}
        expected_pytree = {"items1" : jnp.zeros([3, 3, 4]), "item2" : jnp.zeros([3, 3, 10, 4])}
        new_builder = builder.define_accumulator_by_shape("tree_acc", valid_pytree_shapes)
        self.assertIn("tree_acc", new_builder.defaults)
        self.assertTrue(utils.are_pytrees_equal(new_builder.defaults["tree_acc"], expected_pytree))

        # Define a valid accumulator. Make it be a different dtype

        valid_shape = [3, 3, 3]
        expected_accumulator = jnp.zeros(valid_shape)
        new_builder = builder.define_accumulator_by_shape("valid_acc", valid_shape,
                                                          dtype=jnp.float16)
        self.assertIn("valid_acc", new_builder.defaults)
        self.assertTrue(new_builder.defaults["valid_acc"].dtype == jnp.float16)
        self.assertTrue(jnp.all(new_builder.defaults["valid_acc"] == expected_accumulator))

        # Attempt to define an accumulator with invalid shape (non-int list)
        invalid_shape = {"matrix": [3, "bad", 3]}
        with self.assertRaises(TypeError) as err:
            new_builder.define_accumulator_by_shape("invalid_acc", invalid_shape)
        if SHOW_ERROR_MESSAGES:
            print("Testing define_accumulator_by_shape: Invalid shape definition")
            print(err.exception)

        # Attempt to define an accumulator with invalid dtype
        with self.assertRaises(ValueError) as err:
            new_builder.define_accumulator_by_shape("invalid_acc", valid_shape, dtype=jnp.int32)
        if SHOW_ERROR_MESSAGES:
            print("Testing define_accumulator_by_shape: Invalid dtype")
            print(err.exception)
            print(err.exception.__cause__)
    def test_delete_definition(self):
        """ Test delete_definition method for various scenarios """

        # Initialize a builder for testing
        builder = ControllerBuilder.new_builder(batch_shape=[3, 3])

        # Define a valid accumulator
        valid_accumulator = {"matrix": jnp.zeros([3, 3, 3]), "normalizer": jnp.zeros([3, 3])}
        builder = builder.define_accumulator_directly("valid_acc", valid_accumulator)

        # Delete the defined accumulator
        new_builder = builder.delete_definition("valid_acc")
        self.assertNotIn("valid_acc", new_builder.defaults)
        self.assertNotIn("valid_acc", new_builder.accumulators)
        self.assertNotIn("valid_acc", new_builder.updates)

        # Attempt to delete a non-existing accumulator
        with self.assertRaises(KeyError) as err:
            new_builder.delete_definition("non_existing_acc")
        if SHOW_ERROR_MESSAGES:
            print("Testing delete_definition: Non-existing accumulator")
            print(err.exception)

class test_setters(unittest.TestCase):
    def test_set_probabilities(self):
        """ Test set_probabilities method in ControllerBuilder """

        # Initialize a builder
        builder = ControllerBuilder.new_builder(batch_shape=[3, 3])

        # Set new values for probabilities
        new_probabilities = jnp.ones([3, 3])
        updated_builder = builder.set_probabilities(new_probabilities)
        print(updated_builder.probabilities)
        self.assertTrue(jnp.all(updated_builder.probabilities == new_probabilities))

        # Attempt to set probabilities with values above one (should raise ValueError)
        above_one_probabilities = jnp.full([3, 3], 1.1)  # Values greater than 1
        with self.assertRaises(ValueError) as err:
            builder.set_probabilities(above_one_probabilities)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_probabilities: Values above one")
            print(err.exception)
            print(err.exception.__cause__)

        # Attempt to set probabilities with mismatched shape (should raise ValueError)
        mismatched_shape_probabilities = jnp.ones([4, 4])
        with self.assertRaises(ValueError) as err:
            builder.set_probabilities(mismatched_shape_probabilities)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_probabilities: Mismatched shape")
            print(err.exception)
            print(err.exception.__cause__)

        # Attempt to set probabilities with mismatched dtype (should raise TypeError)
        mismatched_dtype_probabilities = jnp.ones([3, 3], dtype=jnp.int32)
        with self.assertRaises(ValueError) as err:
            builder.set_probabilities(mismatched_dtype_probabilities)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_probabilities: Mismatched dtype")
            print(err.exception)
            print(err.exception.__cause__)
    def test_set_residuals(self):
        """ Test set_residuals method in ControllerBuilder """

        # Initialize a builder
        builder = ControllerBuilder.new_builder(batch_shape=[3, 3])

        # Set new values for residuals
        new_residuals = jnp.ones([3, 3])
        updated_builder = builder.set_residuals(new_residuals)
        self.assertTrue(jnp.all(updated_builder.residuals == new_residuals))

        # Attempt to set residuals with mismatched shape (should raise ValueError)
        mismatched_shape_residuals = jnp.ones([4, 4])
        with self.assertRaises(ValueError) as err:
            builder.set_residuals(mismatched_shape_residuals)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_residuals: Mismatched shape")
            print(err.exception)
            print(err.exception.__cause__)

        # Attempt to set residuals with mismatched dtype (should raise TypeError)
        mismatched_dtype_residuals = jnp.ones([3, 3], dtype=jnp.int32)
        with self.assertRaises(ValueError) as err:
            builder.set_residuals(mismatched_dtype_residuals)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_residuals: Mismatched dtype")
            print(err.exception)
            print(err.exception.__cause__)

        # Attempt to set residuals with values above one (should raise ValueError)
        above_one_residuals = jnp.full([3, 3], 1.1)
        with self.assertRaises(ValueError) as err:
            builder.set_residuals(above_one_residuals)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_residuals: Values above one")
            print(err.exception)
            print(err.exception.__cause__)

        # Attempt to set residuals with values below zero (should raise ValueError)
        below_zero_residuals = jnp.full([3, 3], -0.1)
        with self.assertRaises(ValueError) as err:
            builder.set_residuals(below_zero_residuals)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_residuals: Values below zero")
            print(err.exception)
            print(err.exception.__cause__)
    def test_set_iterations(self):
        """ Test set_iterations method in ControllerBuilder """

        # Initialize a builder
        builder = ControllerBuilder.new_builder(batch_shape=[3, 3])

        # Set new values for iterations (should be int32)
        new_iterations = jnp.array([[1, 2, 3], [4, 5, 6],[7,9,8]], dtype=jnp.int32)
        updated_builder = builder.set_iterations(new_iterations)
        self.assertTrue(jnp.all(updated_builder.iterations == new_iterations))

        # Attempt to set iterations with mismatched shape (should raise ValueError)
        mismatched_shape_iterations = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.int32)
        with self.assertRaises(ValueError) as err:
            builder.set_iterations(mismatched_shape_iterations)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_iterations: Mismatched shape")
            print(err.exception)
            print(err.exception.__cause__)

        # Attempt to set iterations with mismatched dtype (should raise TypeError)
        mismatched_dtype_iterations = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32)
        with self.assertRaises(ValueError) as err:
            builder.set_iterations(mismatched_dtype_iterations)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_iterations: Mismatched dtype")
            print(err.exception)
            print(err.exception.__cause__)

    def test_set_epsilon(self):
        """ Test set_epsilon method in ControllerBuilder """

        # Initialize a builder
        builder = ControllerBuilder.new_builder(batch_shape=[3, 3])

        # Set a valid epsilon value
        new_epsilon = 0.5
        updated_builder = builder.set_epsilon(new_epsilon)
        self.assertEqual(updated_builder.state.epsilon, new_epsilon)

        # Attempt to set epsilon with a non-float value (should raise ValueError)
        non_float_epsilon = "0.5"
        with self.assertRaises(ValueError) as err:
            builder.set_epsilon(non_float_epsilon)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_epsilon: Non-float value")
            print(err.exception)

        # Attempt to set epsilon with a value less than 0 (should raise ValueError)
        below_zero_epsilon = -0.1
        with self.assertRaises(ValueError) as err:
            builder.set_epsilon(below_zero_epsilon)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_epsilon: Value below zero")
            print(err.exception)

        # Attempt to set epsilon with a value greater than 1 (should raise ValueError)
        above_one_epsilon = 1.1
        with self.assertRaises(ValueError) as err:
            builder.set_epsilon(above_one_epsilon)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_epsilon: Value above one")
            print(err.exception)

    def test_set_accumulator(self):
        """ Test set_accumulator method in ControllerBuilder """

        # Initialize a builder and define an accumulator
        builder = ControllerBuilder.new_builder(batch_shape=[3, 3])
        initial_accumulator = {"matrix": jnp.zeros([3, 3, 3]), "normalizer": jnp.zeros([3, 3])}
        builder = builder.define_accumulator_directly("test_acc", initial_accumulator)

        # Set a new value for the defined accumulator
        new_accumulator_value = {"matrix": jnp.ones([3, 3, 3]), "normalizer": jnp.ones([3, 3])}
        updated_builder = builder.set_accumulator("test_acc", new_accumulator_value)
        self.assertTrue(jnp.all(updated_builder.accumulators["test_acc"]["matrix"] == jnp.ones([3, 3, 3])))
        self.assertTrue(jnp.all(updated_builder.accumulators["test_acc"]["normalizer"] == jnp.ones([3, 3])))

        # Attempt to set a non-existing accumulator (should raise KeyError)
        with self.assertRaises(KeyError) as err:
            builder.set_accumulator("non_existing_acc", new_accumulator_value)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_accumulator: Non-existing accumulator")
            print(err.exception)

        # Attempt to set an accumulator with incompatible pytree (should raise ValueError)
        incompatible_accumulator_value = {"matrix": jnp.ones([4, 4, 4]), "normalizer": jnp.ones([3, 3])}
        with self.assertRaises(ValueError) as err:
            builder.set_accumulator("test_acc", incompatible_accumulator_value)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_accumulator: Incompatible accumulator value")
            print(err.exception)
            print(err.exception.__cause__)
    def test_set_defaults(self):
        """ Test set_defaults method in ControllerBuilder """

        # Initialize a builder and define an accumulator with defaults
        builder = ControllerBuilder.new_builder(batch_shape=[3, 3])
        initial_default = {"matrix": jnp.zeros([3, 3, 3]), "normalizer": jnp.zeros([3, 3])}
        builder = builder.define_accumulator_directly("test_acc", initial_default)

        # Set new default values for the defined accumulator
        new_default_value = {"matrix": jnp.ones([3, 3, 3]), "normalizer": jnp.ones([3, 3])}
        updated_builder = builder.set_defaults("test_acc", new_default_value)
        self.assertTrue(jnp.all(updated_builder.defaults["test_acc"]["matrix"] == jnp.ones([3, 3, 3])))
        self.assertTrue(jnp.all(updated_builder.defaults["test_acc"]["normalizer"] == jnp.ones([3, 3])))

        # Attempt to set defaults for a non-existing accumulator (should raise KeyError)
        with self.assertRaises(KeyError) as err:
            builder.set_defaults("non_existing_acc", new_default_value)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_defaults: Non-existing accumulator")
            print(err.exception)

        # Attempt to set defaults with incompatible pytree (should raise ValueError)
        incompatible_default_value = {"matrix": jnp.ones([4, 4, 4]), "normalizer": jnp.ones([3, 3])}
        with self.assertRaises(ValueError) as err:
            builder.set_defaults("test_acc", incompatible_default_value)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_defaults: Incompatible default value")
            print(err.exception)
            print(err.exception.__cause__)
    def test_set_updates(self):
        """ Test set_updates method in ControllerBuilder """

        # Initialize a builder and define an accumulator
        builder = ControllerBuilder.new_builder(batch_shape=[3, 3])
        initial_accumulator = {"matrix": jnp.zeros([3, 3, 3]), "normalizer": jnp.zeros([3, 3])}
        builder = builder.define_accumulator_directly("test_acc", initial_accumulator)

        # Set new update values for the defined accumulator
        new_update_value = {"matrix": jnp.ones([3, 3, 3]), "normalizer": jnp.ones([3, 3])}
        updated_builder = builder.set_updates("test_acc", new_update_value)
        self.assertTrue(jnp.all(updated_builder.updates["test_acc"]["matrix"] == jnp.ones([3, 3, 3])))
        self.assertTrue(jnp.all(updated_builder.updates["test_acc"]["normalizer"] == jnp.ones([3, 3])))

        # Attempt to set updates for a non-existing accumulator (should raise KeyError)
        with self.assertRaises(KeyError) as err:
            builder.set_updates("non_existing_acc", new_update_value)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_updates: Non-existing accumulator")
            print(err.exception)

        # Attempt to set updates with incompatible pytree (should raise ValueError)
        incompatible_update_value = {"matrix": jnp.ones([4, 4, 4]), "normalizer": jnp.ones([3, 3])}
        with self.assertRaises(ValueError) as err:
            builder.set_updates("test_acc", incompatible_update_value)
        if SHOW_ERROR_MESSAGES:
            print("Testing set_updates: Incompatible update value")
            print(err.exception)
            print(err.exception.__cause__)

class test_jittable(unittest.TestCase):
    """
    Tests for jit jax compatibility.
    """

    def test_build_jittable(self):
        """Test if the creation of a builder is jit compatible"""
        batch_shape = 10
        state_shape = [batch_shape, 20]
        output_shape = [batch_shape, 10]

        def build_controller()->ACT_Controller:
            builder = ControllerBuilder.new_builder(batch_shape)
            builder = builder.define_accumulator_by_shape("state", state_shape)
            builder = builder.define_accumulator_by_shape("output", output_shape)
            return builder.build()
        jax.config.update("jax_traceback_filtering", "off")

        jitted_build_creator = jax.jit(build_controller)
        controller = jitted_build_creator()
    def test_properties_jit(self):
        """Test if accessing properties is jit compatible, for all relevant properties"""

        batch_shape = 10
        state_shape = [batch_shape, 20]
        output_shape = [batch_shape, 10]

        builder = ControllerBuilder.new_builder(batch_shape)
        builder = builder.define_accumulator_by_shape("state", state_shape)
        builder = builder.define_accumulator_by_shape("output", output_shape)

        # Define the various getter cases, in terms of uncalled
        # functions. Then call the built jit stup
        epsilon = jax.jit(lambda : builder.epsilon)()
        iterations = jax.jit(lambda : builder.iterations)()
        probabilities = jax.jit(lambda : builder.probabilities)()
        residuals = jax.jit(lambda : builder.residuals)()
        defaults = jax.jit(lambda : builder.defaults)()
        accumulators = jax.jit(lambda : builder.accumulators)()
        updates = jax.jit(lambda  : builder.updates)()



        # Call the

