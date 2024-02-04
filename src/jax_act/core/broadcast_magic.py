"""

A manipulator designed to allow the easy manipulation of
underlying state in terms of probabilities broadcast across
internal tensors.

The general idea is if you have ControllerA, and ControllerB, it would
be nice for downstream arithmetic sometimes to be able to do something like:

0.3*ControllerA + 0.7*ControllerB\

Mathematical Rules of Operation:
    We define a new object, called a BroadcastEditor, which has the following fields.

    1) It contains within it a boolean config, one element per

Initialization and configuration:
    BroadcastEditors are initialize with a configuration that tells you what features of
    the internal state an operand will be replicated across when attempting to apply
    a supported operation.

    For example, with config of only accumulators=True,
    trying to add jnp.ones([2]) will just add that tensor to each element of the pytree in
    the accumulators. This might work, depending on your scenario. Meanwhile, if you try
    try the same thing but with (accumulators=True, epsilon=True) you will end up with a
    tensor of epsilons, which is not exactly valid.


Initialization:
   -
   -
General design:
1) When a BroadcastEditor is created, you specify what of the features of ACTStates will be
   edited. These will be broadcasted into by magic action, and so should be thought about carefully
2) In order to perform __magic__ action between two BroadcastEditor instances, all non-edited features
   must be exactly the same. This will mean that if you do not edit the probabilities involved,
   for instance, they are unlikely to line up allowing you to combine the results. This prevents
   ambiguity: If you have BEditor1, or BEditor2, should you keep the probabities from 1 or 2?
3) If you manually call into a BroadcastEditor operator with another BroadcastEditor, any
1) You may not combine controllers together unless both BroadcastEditing's allow editing of all
internal state features, so clean interpolation is possible.
2) You may, when not
"""

import jax
from jax import numpy as jnp
from src.jax_act.core.types import PyTree
from src.jax_act.core.controller import ACT_Controller
from src.jax_act.core.states import ACTStates
from src.jax_act.core import utils
from typing import Callable, Union, Tuple, Any

PyTree = Union[PyTree, ACTStates]

@dataclass
class

class BroadcastEditing:
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

    def _broadcast_pytrees_to_compatible(operand_a: PyTree,
                                         operand_b: PyTree)->Tuple[PyTree, PyTree]:
        # The pytree with leaves should be broadcast
        # to match the shape of the one with many

        num_a_leaves = len(jax.tree_util.tree_leaves(operand_a))
        num_b_leaves = len(jax.tree_util.tree_leaves(operand_b))
        if num_a_leaves > num_b_leaves:
            # Operand a has more leaves than b. We conclude there is
            # a one-to-many relations between leaves of b and a.
            #
            # We change b to be compatible and have the same tree shape.
            # Then we flesh out any remaining conflicts during left
            # broadcasting
            operand_b = utils.broadcast_pytree_shape(operand_b,target_structure=operand_a)
        else:
            # Operand b has more leaves than a. We conclude there is
            # a one-to-many relations between leaves of a and n.
            #
            # We change a to have the same tree shape and have compatible tensors.
            operand_a = utils.broadcast_pytree_shape(operand_a,target_structure=operand_b)
        return operand_a, operand_b

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
    def __add__(self,
                other: Union[PyTree, 'BroadcastEditing']
                )-> 'BroadcastEditing':
        context_error_message = "Issue encountered during left addition in BroadcastMagic"



        other = other.state if isinstance(other, BroadcastEditing) else other
        new_state = self.execute_binary_operator(jnp.add,
                                                 self.state,
                                                 other,
                                                 context_error_message)
        return BroadcastEditing(new_state)






    def __init__(self, state: ACTStates):
        self.state = state
