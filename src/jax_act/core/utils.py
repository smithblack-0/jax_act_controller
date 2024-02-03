from typing import Callable, Any, List, Optional, Tuple

import jax
import textwrap
import numpy as np

from jax import numpy as jnp
from jax.tree_util import PyTreeDef
from src.jax_act.core.types import PyTree

def format_error_message(message: str, context: str)->str:
    """
    Formats an error message to look nice, with context
    placed first, and then the message shown at an indent.

    :param message: The message to display
    :param context: The context for the message
    :return: A string representing the message
    """
    context = textwrap.dedent(context)
    message = textwrap.dedent(message)
    message = textwrap.indent(message, "    ")
    return context + "\n" + message

def setup_left_broadcast(tensor: jnp.ndarray,
                          target: jnp.ndarray
                          ) -> jnp.ndarray:
    """
    Sets up tensor by unsqueezing dimensions for
    a left broadcast with the target.

    Returns the unsqueezed tensor.

    :param tensor: The tensor to expand
    :param target: The target whose length to match
    :return: The unsqueezed tensor
    """

    assert len(tensor.shape) <= len(target.shape)
    while len(tensor.shape) < len(target.shape):
        tensor = tensor[..., None]
    return tensor


def merge_pytrees(function: Callable[[Any,
                                      Any],
                                      Any],
                   primary_tree: PyTree,
                   auxilary_tree: PyTree,
                   ) -> PyTree:
    """
    Used to merge two pytrees together.

    This deterministically walks the primary and auxilary
    trees, then calls function when like leaves are reached. The
    results are created into a new tree.

    :param function: A function that accepts first a primary leaf, then a auxilary leaf
    :param primary_tree: The primary tree to draw from
    :param auxilary_tree: The auxilary tree to draw from
    :return: A new PyTree
    """
    # Merging PyTrees turns out to be more difficult than
    # expected.
    #
    # Jax, strangly, has no multitree map, which means walking through the
    # nodes in parallel across multiple trees is not possible. Instead, we flatten
    # both trees deterministically, walk through the leaves, and then restore the
    # original structure.

    treedef = jax.tree_util.tree_structure(primary_tree)
    primary_leaves = jax.tree_util.tree_flatten(primary_tree)[0]
    auxilary_leaves = jax.tree_util.tree_flatten(auxilary_tree)[0]

    assert len(primary_leaves) == len(auxilary_leaves)  # Just a quick sanity check.

    new_leaves = []
    for primary_leaf, aux_leaf in zip(primary_leaves, auxilary_leaves):
        update = function(primary_leaf, aux_leaf)
        new_leaves.append(update)

    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def are_pytree_structure_equal(tree_one: PyTree, tree_two: PyTree)->bool:
    """
    Checks if two pytrees are defined using the same tree structure. Ignores the
    contents of the leaves.

    :param tree_one: The first tree to compare
    :param tree_two: The second tree to compare.
    :return: A bool, whether the pytrees are the same
    """
    # It turns out in some cases pytree definitions store information
    # about the nodes in their tree definitions. This is a weird choice, but
    # whatever. Our first task is thus o ensure all nodes are filled with the same value

    tree_one = jax.tree_util.tree_map(lambda x : None, tree_one)
    tree_two = jax.tree_util.tree_map(lambda x : None, tree_two)

    # We compare the tree definitions.

    treedef_one = jax.tree_util.tree_structure(tree_one)
    treedef_two = jax.tree_util.tree_structure(tree_two)
    return treedef_one == treedef_two


def are_pytrees_equal(tree_one: PyTree, tree_two: PyTree, use_allclose: bool = True)->bool:
    """
    Checks if two pytrees are almost equal to one another.
    :param tree_one:
    :param tree_two:
     :return:
    """

    def are_leaves_equal(leaf_one: Any, leaf_two: Any)->bool:
        if type(leaf_one) != type(leaf_two):
            return False
        elif isinstance(leaf_one, (jnp.ndarray, np.ndarray)):
            if use_allclose:
                return jnp.allclose(leaf_one, leaf_two)
            else:
                return jnp.all(leaf_one == leaf_two)
        else:
            return leaf_one == leaf_two

    result_tree = merge_pytrees(are_leaves_equal, tree_one, tree_two)
    leaves = jax.tree_util.tree_flatten(result_tree)[0]
    return all(leaves)

@dataclass
class node_view:
    node_position: int
    num_children: int
def _traverse_pytree(source: PyTreeDef,
                     target: PyTree,
                     )->:
    """
    Objective is to return where you saw
    :param source:
    :param target:
    :return:
    """
    if jax.tree_util.treedef_is_leaf(source) and jax.tree_util.treedef_is_leaf(target):
        return node_view(0, 1)

class node_repeater:
    def __init__(self,
                 source_leaves: List[Any],
                 output: Optional[List[Any]] = None
                 ):
        if output is None:
            output = []

        self.source_leaves = source_leaves
        self.output = output
        self.node = 0
    def repeat_source(self, num_times: int):


def _repeat_node(source_leaf: jnp.ndarray,
                num_times: int
                )->List[jnp.ndarray]:
    return [source_leaf]*num_times

def _count_linked_leaves(treedef: PyTreeDef)->int:
    """
    A depth-first traversal to count
    the number of children linked to the node
    in the given treedef

    :param treedef: The treedef to count
    :return: An integer representing the count
    """
    child_nodes = jax.tree_util.treedef_children(treedef)
    leaves = 0
    for child_node in child_nodes:
        if jax.tree_util.treedef_is_leaf(child_node):
            leaves += 1
        else:
            leaves += _count_linked_leaves(child_node)
    return leaves

def _replicate_leaves(source_treedef: PyTreeDef,
                      target_treedef: PyTreeDef,
                      leaves: List[jnp.ndarray],
                      context_message: str,
                      )->Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """
    A depth-first recursive traversal of the source and
    target trees in a function.

    It promises to return a leaves list in which the leaves
    that were replicated have been replicated. It also returns
    an integer, needed internally, that says how many elements
    out of the leaf reservour were consumed to do it.

    :param source_treedef: The tree_def from the source tree at the current node
    :param target_treedef: The tree_def from the target tree at the current node
    :param source_leaf_reservour: The current leaves we have to draw on
    :return:
        - A list of arrays, representing properly replicated leaves ready for broadcast
        - A list of arrays, representing what leaves are currently left over. Should be used to update
          the internal leaves array
    """
    # Broadcasting between tensors is uninteresting and covered in a million other places
    #
    # We talk here about broadcasting between pytrees. To broadcast between two pytrees,
    # you should essentially walk both trees and track down where there is a leaf on the
    # source tree, and find out what that leaf is. Then, continue walking the target tree,
    # and everywhere there is a leaf in it replicate the current source leaf.
    #
    # This is all well and good, and if the tools supported it this is how we would do
    # this. However, the API is not so nice. Instead, we do something that effectively behaves
    # the same way.
    #


    if jax.tree_util.treedef_is_leaf(source_treedef):

        num_times_to_repeat = _count_linked_leaves(target_treedef)
        return _repeat_node(leaves[0], num_times_to_repeat), leaves[1:]
    elif jax.tree_util.treedef_is_leaf(target_treedef):
        msg = """
        Source tree was not broadcastable.
         
        Source tree had pytree branch node, at position target
        tree had tensor node. This is not allowed.
        
        Did you swap the order you gave your parameters?
        """
        msg = format_error_message(msg, context_message)
        raise RuntimeError(msg)

    source_children = jax.tree_util.treedef_children(source_treedef)
    target_children = jax.tree_util.treedef_children(target_treedef)
    if len(source_children) != len(target_children):
        msg = """
        Source tree was not broadcastable with target tree
        
        Source tree and target tree have incompatible shapes. Despite
        not being a tensor node, the number of children were different.
        
        This is not allowed.
        """
        msg = format_error_message(msg, context_message)
        raise RuntimeError(msg)

    output = []
    for i, (source_node, target_node) in enumerate(zip(source_children, target_children)):
        try:
            update, leaves = _replicate_leaves(source_node, target_node, leaves, context_message)
            output += update
        except Exception as err:
            msg = f"""
            Issue occurred between trees on child {i} of the following
            pytrees.
            """
            msg = textwrap.dedent(msg)
            msg += "\nOn source pytree: \n{%s} \n" % source_treedef
            msg += "\nOn target pytree: \n{%s}" % target_treedef
            msg = format_error_message(msg, context_message)
            raise RuntimeError(msg) from err
    return output, leaves

def can_right_broadcast(array_a: jnp.ndarray,
                        array_b: jnp.ndarray
                        )->bool:
    """
    Test whether or not source array can be broadcast with
    target array. Return bool.

    Broadcasting considers dimensions when tensors are aligned with dimensions
    to the right. In this configuration, the tensors are broadcastable if,
    for each overlapping dimensions:
        1) The dimensions are equal
        2) Or one of them is equal to one

    See numpy broadcasting for details.

    :param array_a: Array we will try to broadcast
    :param array_b: Array we need to be able to broadcast to
    :return: a bool
    """

    broadcast_overlap = jnp.min(array_a.ndim, array_b.ndim)
    if broadcast_overlap == 0:
        # Scalar arrays can always be broadcast
        return True

    # Both arrays are at least 1d. Check overlapping region
    relevant_a_shape = array_a.shape[-broadcast_overlap:]
    relevant_b_shape = array_b.shape[-broadcast_overlap:]
    for a_dim, b_dim in zip(relevant_a_shape, relevant_b_shape):
        if (a_dim == 1) or (b_dim == 1):
            continue
        if a_dim == b_dim:
            continue
        return False
    return True

def setup_broadcast_pytree(source: PyTree,
                         target: PyTree,
                         broadcast_mode: str,
                         )->PyTree:
    """
    A function for broadcasting pytrees such
    that their shapes become compatible.

    This operates similar to normal broadcasting, and indeed sets up
    broadcasting by similar rules. However, there is a catch - it may
    also broadcast nodes. In particular, if during synchronous traversal
    the algorithm sees a tensor leaf on the source tree, and a further pytree
    on the target structure, it will attempt to transform the source tensor
    structure to be compatible with the target structure and repeat that structure

    As a simple example, consider a pair of trees consisting of tree_a, just a tensor,
     and tree_b, a list of tensors. This would be the following

    tree_a = tensor_1a
    tree_b = [tensor_1b,tensor_2b]

    If we perform broadcast_pytree(tree_a, target=tree_b), the result will end
    up being like:

    output = [broadcast_to_match(tensor_1a, target = tensor_1b,
              broadcast_to_match(tensor_1a, target=tensor_2b)
             ]

    Notice the PyTree structure at the end now matches tree_b, with the node in that tree
    determining exactly how we are broadcasting!

    :param source: The tree to start from
    :param target: The tree to try to broadcast to
    :param broadcast_mode: 'Left' or 'Right'
    :return: A broadcast tree
    """

    # We use jax.tree_utils.tree_unflatten to get this job done.
    #
    # Context for this is that jax.tree_util.flatten uses depth
    # first traversal when flattening, and the nodes are
    # manipulated while flat.
    #
    # Nodes must be approached in two ways to make that work. First,
    # each source node is compared to the state of the tree of the
    # target pytree at the same location. We track down how many children
    # are attached to the node at target, then repeat the source node
    # that many times. This ensures one source node is corrolated
    # for each target node
    #
    # After this, we simply broadcast the two

    error_context = "An issue occurred while trying to broadcast two pytrees together: "
    source_leaves, source_treedef = jax.tree_util.tree_flatten(source)
    target_leaves, target_treedef = jax.tree_util.tree_flatten(target)

    # Replicates source leaves when one source leaf corresponds to multiple leaves in the
    # target treedef. This ensures the number of leaves is correct.
    source_leaves, _ = _replicate_leaves(source_treedef, target_treedef, source_leaves, error_context)

    # Broadcast each leaf to be compatible.
    assert len(source_leaves) == len(target_leaves), "error: 1 This should never happen, yell at maintainer"

    final_leaves = []
    for i, (source_leaf, target_leaf) in enumerate(zip(source_leaves, target_leaves)):
        if broadcast_mode == "Left":
            update = setup_left_broadcast(source_leaf, target_leaf)
        else:
            if not can_right_broadcast(source_leaf, target_leaf):
                msg = f"""
                Source leaf and target leaf could not be broadcast
                together. This concerned target leaf {i}, and 
                nodes of shape 
                
                {source_leaf.shape}, and {target_leaf.shape}
                """
                msg = format_error_message(msg, error_context)
                raise RuntimeError(msg)
            update = source_leaf
        final_leaves.append(update)
    return jax.tree_util.tree_unflatten(target_treedef, final_leaves)


