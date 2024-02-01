"""
Tests for the builder act mechanism
"""

import itertools
import unittest
import warnings
import jax

from jax import numpy as jnp
from jax.experimental import checkify
from src.jax_act.builder import ControllerBuilder
from src.jax_act.states import ACTStates
from src.jax_act import utils
from src.jax_act.controller import ACT_Controller

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

class test_validation(unittest.TestCase):
    """ Test that the validation functions operate properly"""

    def test_validate_definition_pytree(self):
        """ Test if _validate_definition_pytree throws errors appropriately """

        batch_shape = [3, 2]
        good_tensor = jnp.zeros(batch_shape, dtype=jnp.float32)
        bad_dtype_tensor = jnp.zeros(batch_shape, dtype=jnp.int32)
        bad_shape_tensor = jnp.zeros((5, 3), dtype=jnp.float32)
        bad_length_tensor = jnp.zeros([2], dtype=jnp.float32)

        good_tree = {"leaf1": good_tensor, "leaf2": good_tensor}
        bad_dtype_tree = {"leaf1": good_tensor, "leaf2": bad_dtype_tensor}
        bad_length_tree = {"leaf1" : good_tensor, "leaf2" : bad_length_tensor}
        bad_shape_tree = {"leaf1": good_tensor, "leaf2": bad_shape_tensor}

        builder = ControllerBuilder.new_builder(batch_shape)  # Initialize with appropriate parameters
        info_msg = "Testing validation of pytrees"


        # Test that valid pytree does not throw
        builder._validate_definition_pytree(good_tree, info_msg)
        jit_validation = jax.jit( builder._validate_definition_pytree, static_argnums=[1])
        jit_validation(good_tree, info_msg)

        # Test for invalid tensor dtype
        with self.assertRaises(TypeError) as err:
            builder._validate_definition_pytree(bad_dtype_tree, info_msg)
        if SHOW_ERROR_MESSAGES:
            print("Testing _validate_definition_pytree: Invalid Tensor Dtype")
            print(err.exception)

        # Test for invalid tensor dtype with jit
        with self.assertRaises(TypeError) as err:
            jit_validation(bad_dtype_tree, info_msg)
        if SHOW_ERROR_MESSAGES:
            print("Testing _validate_definition_pytree with jit: Invalid Tensor Dtype")
            print(err.exception)

        # Test for invalid batch length
        with self.assertRaises(ValueError) as err:
            builder._validate_definition_pytree(bad_length_tree, info_msg)
        if SHOW_ERROR_MESSAGES:
            print("Testing _validate_definition_pytree with jit: Invalid batch length")
            print(err.exception)

        # Test for invalid batch length under jit
        with self.assertRaises(ValueError) as err:
            jit_validation(bad_length_tree, info_msg)
        if SHOW_ERROR_MESSAGES:
            print("Testing _validate_definition_pytree with jit: Invalid batch length")
            print(err.exception)

        # Test for invalid tensor shape
        with self.assertRaises(ValueError) as err:
            builder._validate_definition_pytree(bad_shape_tree, info_msg)
        if SHOW_ERROR_MESSAGES:
            print("Testing _validate_definition_pytree: Invalid Tensor Shape")
            print(err.exception)

        # Test for invalid tensor shape under jit
        with self.assertRaises(ValueError) as err:
            jit_validation(bad_shape_tree, info_msg)
        if SHOW_ERROR_MESSAGES:
            print("Testing _validate_definition_pytree with jit: Invalid Tensor Shape")
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

        jit_validation = jax.jit(ControllerBuilder.new_builder, static_argnums=[1, 2])
        # Case 1: core_dtype is not a floating type
        with self.assertRaises(ValueError) as err:
            ControllerBuilder.new_builder(batch_shape=[3, 3], core_dtype=jnp.int32)
        if SHOW_ERROR_MESSAGES:
            print("Testing new_builder: core_dtype not floating")
            print(err.exception)

        # Case 1: core_dtype is not a floating type, jit
        with self.assertRaises(ValueError) as err:
            jit_validation(batch_shape=[3, 3], core_dtype=jnp.int32)
        if SHOW_ERROR_MESSAGES:
            print("Testing new_builder with jit: core_dtype not floating")
            print(err.exception)

        # Case 2: epsilon is not between 0 and 1 (testing with a value greater than 1)
        with self.assertRaises(ValueError) as err:
            ControllerBuilder.new_builder(batch_shape=[3, 3], epsilon=1.5)
        if SHOW_ERROR_MESSAGES:
            print("Testing new_builder: epsilon greater than 1")
            print(err.exception)

        # Case 2: epsilon is not between 0 and 1 jit(testing with a value greater than 1)
        with self.assertRaises(ValueError) as err:
            jit_validation(batch_shape=[3, 3], epsilon=1.5)
        if SHOW_ERROR_MESSAGES:
            print("Testing new_builder with jit: epsilon greater than 1")
            print(err.exception)

        # Case 3: epsilon is not between 0 and 1 (testing with a value less than 0)
        with self.assertRaises(ValueError) as err:
            ControllerBuilder.new_builder(batch_shape=[3, 3], epsilon=-0.1)
        if SHOW_ERROR_MESSAGES:
            print("Testing new_builder: epsilon less than 0")
            print(err.exception)

        # Case 3: epsilon is not between 0 and 1 with jit (testing with a value less than 0)
        with self.assertRaises(ValueError) as err:
            jit_validation(batch_shape=[3, 3], epsilon=-0.1)
        if SHOW_ERROR_MESSAGES:
            print("Testing new_builder with jit: epsilon less than 0")
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
        context_error_message = "The issue occurred while testing define private"

        # Test new definition
        definition = {"items1" : tensor, "items2" : tensor_two}

        def define():
            builder = ControllerBuilder.new_builder(batch_shape)
            builder = builder._define("test_accumulator", definition, context_error_message)
            return builder

        builder = define()

        self.assertTrue(utils.are_pytrees_equal(builder.accumulators["test_accumulator"], definition))
        self.assertTrue(utils.are_pytrees_equal(builder.defaults["test_accumulator"], definition))
        self.assertTrue(builder.updates["test_accumulator"] is None)

        # Test new definition with jit
        define = jax.jit(define)
        builder = define()

        self.assertTrue(utils.are_pytrees_equal(builder.accumulators["test_accumulator"], definition))
        self.assertTrue(utils.are_pytrees_equal(builder.defaults["test_accumulator"], definition))
        self.assertTrue(builder.updates["test_accumulator"] is None)

    def test_define_accumulator_directly(self):
        """ Test define_accumulator_directly method for various scenarios """

        # Initialize a builder for testing
        builder = ControllerBuilder.new_builder(batch_shape=[3, 3])

        # Define a valid accumulator
        def define():
            valid_accumulator = {"matrix": jnp.zeros([3, 3, 3]), "normalizer": jnp.zeros([3, 3])}
            new_builder = builder.define_accumulator_directly("valid_acc", valid_accumulator)
            return new_builder
        new_builder = define()

        self.assertIn("valid_acc", new_builder.defaults)
        self.assertIn("valid_acc", new_builder.accumulators)

        # Define under jit conditions
        define = jax.jit(define)
        new_builder = define()

        self.assertIn("valid_acc", new_builder.defaults)
        self.assertIn("valid_acc", new_builder.accumulators)

        # Attempt to define an accumulator with an invalid definition
        def define():
            invalid_accumulator = {"matrix": [[1, 2, 3]], "normalizer": jnp.zeros([3, 3])}
            new_builder = builder.define_accumulator_directly("invalid_acc", invalid_accumulator)
            return new_builder

        with self.assertRaises(TypeError) as err:
           define()
        if SHOW_ERROR_MESSAGES:
            print("Testing define_accumulator_directly: Invalid accumulator definition")
            print(err.exception)
            print(err.exception.__cause__)

    def test_define_accumulator_by_shape(self):
        """ Test define_accumulator_by_shape method for various scenarios """


        # Define a valid accumulator by shape

        valid_shape = [3, 3, 3]
        expected_accumulator = jnp.zeros(valid_shape)
        def define():
            builder = ControllerBuilder.new_builder(batch_shape=[3, 3])
            new_builder = builder.define_accumulator_by_shape("valid_acc", valid_shape)
            return new_builder

        new_builder = define()
        self.assertIn("valid_acc", new_builder.defaults)
        self.assertTrue(jnp.all(new_builder.accumulators["valid_acc"] == expected_accumulator))

        # Define a valid accumulator, using jit
        define = jax.jit(define)
        new_builder = define()
        self.assertIn("valid_acc", new_builder.defaults)
        self.assertTrue(jnp.all(new_builder.accumulators["valid_acc"] == expected_accumulator))

        # Define a valid accumulator, but this time use a pytree to do it

        valid_pytree_shapes = {"items1" : [3, 3, 4], "item2" : [3, 3, 10, 4]}
        expected_pytree = {"items1" : jnp.zeros([3, 3, 4]), "item2" : jnp.zeros([3, 3, 10, 4])}
        def define():
            builder = ControllerBuilder.new_builder(batch_shape=[3, 3])
            new_builder = builder.define_accumulator_by_shape("tree_acc", valid_pytree_shapes)
            return new_builder

        new_builder = define()
        self.assertIn("tree_acc", new_builder.defaults)
        self.assertTrue(utils.are_pytrees_equal(new_builder.defaults["tree_acc"], expected_pytree))

        define = jax.jit(define)
        new_builder = define()
        self.assertIn("tree_acc", new_builder.defaults)
        self.assertTrue(utils.are_pytrees_equal(new_builder.defaults["tree_acc"], expected_pytree))

        # Define a valid accumulator. Make it be a different dtype

        valid_shape = [3, 3, 3]
        expected_accumulator = jnp.zeros(valid_shape)
        def define():
            builder = ControllerBuilder.new_builder(batch_shape=[3, 3])
            new_builder = builder.define_accumulator_by_shape("valid_acc", valid_shape,
                                                          dtype=jnp.float16)
            return new_builder

        new_builder = define()
        self.assertIn("valid_acc", new_builder.defaults)
        self.assertTrue(new_builder.defaults["valid_acc"].dtype == jnp.float16)
        self.assertTrue(jnp.all(new_builder.defaults["valid_acc"] == expected_accumulator))

        define = jax.jit(define)
        new_builder = define()

        self.assertIn("valid_acc", new_builder.defaults)
        self.assertTrue(new_builder.defaults["valid_acc"].dtype == jnp.float16)
        self.assertTrue(jnp.all(new_builder.defaults["valid_acc"] == expected_accumulator))

        # Attempt to define an accumulator with invalid shape (non-int list)
        invalid_shape = {"matrix": [3, "bad", 3]}
        def define():
            builder = ControllerBuilder.new_builder(batch_shape=[3, 3])
            new_builder = builder.define_accumulator_by_shape("invalid_shape", invalid_shape,
                                                          dtype=jnp.float16)
            return new_builder

        with self.assertRaises(ValueError) as err:
            define()
        if SHOW_ERROR_MESSAGES:
            print("Testing define_accumulator_by_shape: Invalid shape definition")
            print(err.exception)

        define = jax.jit(define)
        with self.assertRaises(ValueError) as err:
            define()
        if SHOW_ERROR_MESSAGES:
            print("Testing define_accumulator_by_shape with jit: Invalid shape definition")
            print(err.exception)

        # Attempt to define an accumulator with invalid dtype
        def define():
            builder = ControllerBuilder.new_builder(batch_shape=[3, 3])
            new_builder = builder.define_accumulator_by_shape("invalid_dtype", valid_shape,
                                                          dtype=jnp.int32)
            return new_builder


        with self.assertRaises(TypeError) as err:
            define()
        if SHOW_ERROR_MESSAGES:
            print("Testing define_accumulator_by_shape: Invalid dtype")
            print(err.exception)
            print(err.exception.__cause__)

        define = jax.jit(define)
        with self.assertRaises(TypeError) as err:
            define()
        if SHOW_ERROR_MESSAGES:
            print("Testing define_accumulator_by_shape: Invalid dtype with jit")
            print(err.exception)
            print(err.exception.__cause__)