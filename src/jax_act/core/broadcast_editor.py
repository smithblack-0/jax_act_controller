"""

A manipulator designed to allow the easy manipulation of
underlying state in terms of probabilities broadcast across
internal tensors.


Mathematical Ideal:
    We define a new operand object, called a BroadcastEditor, which has the following fields.

    1) A field called states containing an ACTStates. This will be updated
    2) A field called config containing a sequence of boolean entries, one per element
       of states.

    An underlying act feature is said to be "on" or being "edited" when
    it's corresponding configuration action is on. Operators are updated
    with regards to this operand object to function as follows:

    - An operation between a tensor and a BroadcastEditor is equal to distributing
      the operation onto all active config states, executing against all tensor collections,
      then rebuilding the broadcast editor
    - An operation between a tensor collection, or "PyTree", is equal to attempting
      to broadcast the pytree between all active config states.
    - An operation between a BroadcastEditor and a BroadcastEditor is only possible
      so long as the configs are identical and the inactive states are identical.
    - All operations return a new instance of BroadcastEditor with the updates applied.

    Additionally, do not expect the class to yell at you if you are inverting your probabilities
    either.
"""

import jax
from jax import numpy as jnp
from src.jax_act.core.types import PyTree
from src.jax_act.core.controller import ACT_Controller
from src.jax_act.core.states import ACTStates, EditorConfig
from src.jax_act.core import utils
from typing import Callable, Union, Tuple, Any
from dataclasses import dataclass
PyTree = Union[PyTree, ACTStates]


class BroadcastEditor:
    """
    Allows for the easy manipulate of the entire
    controller state by probabilities.


    """
    # TODO:
    #    - Properties passthroughs
    #    - Methods:
    #        - Main binary operator
    #        - Main unitary operator
    #        - Various arithmetic methods
    #        - Various magic methods
    #    - Docstring
    @classmethod
    def execute_binary_operator(cls,
                                operator: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
                                operand_a: PyTree,
                                operand_b: PyTree,
                                context_msg: str
                                ) -> PyTree:
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

        try:
            operand_a, operand_b = cls._broadcast_pytrees_to_compatible(operand_a, operand_b)
        except Exception as err:
            msg = """
            An issue occurred while setting up the pytrees composing
            the two operands to be compatible.
            """
            msg = utils.format_error_message(msg, context_msg)
            raise RuntimeError(msg) from err

        tree_def = jax.tree_util.tree_structure(operand_a)
        leaves_on_a = jax.tree_util.tree_leaves(operand_a)
        leaves_on_b = jax.tree_util.tree_leaves(operand_b)

        output = []
        for i, (leaf_a, leaf_b) in enumerate(zip(leaves_on_a, leaves_on_b)):

            try:
                # We need to make sure left broadcast works properly.
                #
                # To do this we find the tensor with the most dimensions, and
                # attempt to broadcast to match
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
    def execute_unitary_operation(operator: Callable[[jnp.ndarray], jnp.ndarray],
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
    def transform(self):
    def __init__(self,
                 state: ACTStates,
                 edit_config: EditorConfig):
        self.state = state
        self.config = config
