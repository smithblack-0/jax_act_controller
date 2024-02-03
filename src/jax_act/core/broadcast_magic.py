"""

A manipulator designed to allow the easy manipulation of
underlying state in terms of probabilities broadcast across
internal tensors.

The general idea is if you have ControllerA, and ControllerB, it would
be nice for downstream arithmetic sometimes to be able to do:

0.3*ControllerA + 0.7*ControllerB
"""

import jax
from jax import numpy as jnp
from src.jax_act.core.types import PyTree
from src.jax_act.core.controller import ACT_Controller
from src.jax_act.core.states import ACTStates
from src.jax_act.core import utils
from typing import Callable, Union

PyTree = Union[PyTree, ACTStates]

class BroadcastMagic:
    # TODO:
    #    - Properties passthroughs
    #    - Methods:
    #        - Main binary operator
    #        - Main unitary operator
    #        - Various arithmetic methods
    #        - Various magic methods
    #    - Docstring

    def __add__(self,
                other: 'BroadcastMagic'
                )->'BroadcastMagic':
        context_error_message = "Issue encountered during left addition in BroadcastMagic"
        new_state = self.perform_binary_operation(jnp.add,
                                                  self.state,
                                                  other.state,
                                                  context_error_message)
        return BroadcastMagic(new_state)

    def __radd__(self,
                 other: 'BroadcastMagic'
                 )->'BroadcastMagic':
        context_error_message = "Issue was encountered during right addition in BroadcastMagic"
        new_state = self.perform_binary_operation(jnp.add,
                                                  self.state,
                                                  other.state,
                                                  context_error_message)
        return BroadcastMagic(new_state)

    def __sub__(self,
                other:
                )->'BroadcastMagic':
        context_error_message = "Issue was encountered during left subtraction in BroadcastMagic"
        new_state = self.perform_binary_operation(jnp.subtract,
                                                  self.state,
                                                  other.state
                                                  context_error_message)
        return BroadcastMagic(new_state)

    def __rsub__(self,
                 other: 'BroadcastMagic'
                 )->'BroadcastMagic':
        context_error_message = "Issue was encountered during right subtraction in BroadcastMagic"
        new_state = self.perform_binary_operation(jnp.subtract,
                                                  other.state,
                                                  self.state,
                                                  context_error_message)
        return BroadcastMagic(new_state)

    def __mul__(self,
                other):

    @staticmethod
    def perform_binary_operation(operator: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                                 operand_a: PyTree,
                                 operand_b: PyTree,
                                 context_msg: str
                                 )->PyTree:
        """
        Performs a binary operation across two pytrees with same treedef
        using an operator defined in terms of tensors and the two trees.

        Importantly, we also promise to left broadcast where needed,
        rather than the traditional right broadcast. This turns out to
        work much better when dealing with probabilities.

        :param operator: The operator to apply.
        :param operand_a: A PyTree with the same shape as b
        :param operand_b: A PyTree with the same shape as a
        :return: A pytree shaped like a and b, where every leaf is the result of
                 putting leaf_a and leaf_b through the operator
        """
        tree_def = jax.tree_util.tree_structure(operand_a)
        leaves_on_a = jax.tree_util.tree_leaves(operand_a)
        leaves_on_b = jax.tree_util.tree_leaves(operand_b)
        if len(leaves_on_a) != len(leaves_on_b):
            msg = """
            An issue was encountered while executing a binary
            operation. It was expected the operations were to be performed
            on pytrees of the same shape, but the number of leaves were different
            """
            msg = utils.format_error_message(msg, context_msg)
            raise RuntimeError(msg)

        output = []
        for i, (leaf_a, leaf_b) in enumerate(zip(leaves_on_a, leaves_on_b)):

            try:
                # We need to make sure left broadcast works properly.
                #
                # To do this we find the tensor with the most dimensions, and
                # broadcast to match
                if leaf_a.ndim > leaf_b.ndim:
                    leaf_b = utils.setup_left_broadcast(leaf_b, target=leaf_a)
                elif leaf_a.ndim < leaf_b.ndim:
                    leaf_a = utils.setup_left_broadcast(leaf_a, target=leaf_b)
                result = operator(leaf_a, leaf_b)

            except Exception as err:
                msg = f"""
                An issue occurred while trying to execute the operator. It failed
                while attempting to execute the operators on leafs '{i}'
                """
                msg = utils.format_error_message(msg, context_msg)
                raise RuntimeError(msg) from err
            output.append(result)

        return jax.tree_util.tree_unflatten(tree_def, output)

    @staticmethod
    def perform_unitary_operation(operator: Callable[[jnp.ndarray], jnp.ndarray],
                                  operand: PyTree,
                                  context_message
                                  )->PyTree:
        """
        Applies a unitary operator onto the operand with rigorous error
        handling

        :param operator: The operator to apply. A callable that accepts and returns a tensor
        :param operand: The operand to apply onto. A PyTree, which will be walked over
        :return: A PyTree whose leaves are the result of applying operator to each leaf in operand
        """
        flat_leaves, tree_def = jax.tree_util.tree_flatten(operand)
        output = []
        for i, leaf in enumerate(flat_leaves):
            try:
                result = operator(leaf)
            except Exception as err:
                msg = f"""
                An issue occurred while applying the operator to 
                leaf {i} of the PyTree structure
                """
                msg = utils.format_error_message(msg, context_message)
                raise RuntimeError(msg) from err
            output.append(result)
        return jax.tree_util.tree_unflatten(tree_def, output)





    def __init__(self, state: ACTStates):
        self.state = state
