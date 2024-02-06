from typing import Callable, Any, List, Optional, Tuple

import jax
import textwrap
import numpy as np

from jax import numpy as jnp
from jax.experimental import checkify
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

def jit_and_checkify(function: Callable)->Callable:
    """
    A utility wrapper for working with the jax
    checkification library, this will catch,
    and raise like normal, checkification errors.

    Hopefully
    """
    checkified_function = checkify.checkify(function)
    jitted_function = jax.jit(checkified_function)
    def checkify_wrapper(*args, **kwargs):
        errors, output = jitted_function(*args, **kwargs)
        errors.throw()
        return output
    return checkify_wrapper

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
    if tensor.ndim == 0:
        # Scalars will always left broadcast just fine
        return tensor

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
    trees, then calls function when like leaves a
    re reached. The
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
def _repeat_node(source_leaf: jnp.ndarray,
                num_times: int
                )->List[jnp.ndarray]:
    return [source_leaf]*num_times

def _is_empty_collection(treedef: PyTreeDef)->bool:
    # The only way I could find to tell apart empty vs nonempty
    # leaf collections was to use treedef.node_data. If
    # empty, the collection class is returned as well. Otherwise,
    # just the data.
    #
    # I would really love to see an API function.
    return isinstance(treedef.node_data(), tuple)
def _count_linked_leaves(treedef: PyTreeDef)->int:
    """
    A depth-first traversal to count
    the number of children linked to the node
    in the given treedef

    :param treedef: The treedef to count
    :return: An integer representing the count
    """

    # Base case. If this IS a leaf, we must make a
    # make a return.
    #
    # Sometimes, an empty collection will be detected as a
    # leaf while contributing nothing when flattening. We handle
    # this using special logic.
    if jax.tree_util.treedef_is_leaf(treedef):
        if _is_empty_collection(treedef):
            return 0
        else:
            return 1

    # This is a branch. Examine its children, in a depth first
    # traversal. Count how many are on each child, gather those
    # together, and return.
    child_nodes = jax.tree_util.treedef_children(treedef)
    leaves = 0
    for child_node in child_nodes:
        leaves += _count_linked_leaves(child_node)
    return leaves

def _validate_children_descent(source_treenode: PyTreeDef,
                               target_treenode: PyTreeDef,
                               context_string: str
                               ):
    """
    A source and target treenode pair are capable of further
    tree descent if and only if for all immediate children

    1) The class of the collections holding entries in the source and target node match
    2) All keys, if they exist, match
    3) The length of the elements match

    :param source_treenode: The proposed source node whose children to descend
    :param target_treenode: The proposed target node whose children to descend
    :raises RuntimeError: If a condition is violated
    """

    source_class, source_keys = source_treenode.node_data()
    target_class, target_keys = target_treenode.node_data()

    source_children = source_treenode.children()
    target_children = target_treenode.children()
    if source_class is not target_class:
        msg = f"""
        Two trees could not be broadcast together due to mismatched 
        tree shape. 
        
        The source tree had a node of type {source_class}.
        However, the target tree had a node of type {target_class} in the 
        same place.
        
        This is not allowed. Make sure nodes either are the same class,
        or one is a tensor and the other a pytree.
        """
        msg = format_error_message(msg, context_string)
        raise RuntimeError(msg)
    if len(source_children) != len(target_children):
        msg = f"""
        Two trees could not be broadcast together due to mismatched 
        tree shape. 
        
        The source tree had {len(source_children)} children attached to the 
        current node. However, at the same place at the target
        tree, there were {len(target_children)} children.
        
        This mismatch is not allowed. Make sure nodes either have
        the same number of children, or one is a tensor and the other
        a pytree.
        """
        msg = format_error_message(msg, context_string)
        raise RuntimeError(msg)
    if source_keys is not None:
        # We are working with a node that has keys, like a dictionary
        #
        # We know at this point source and target keys will be the same length,
        # and are from the same kind of object
        for source_key, target_key in zip(source_keys, target_keys):
            if source_key != target_key:
                msg = f"""
                Two trees could not be broadcast together due to mismatched 
                tree shape. 
                
                A key holding a child on the source node, named '{source_key}',
                was not also found in the same position on target node, instead found
                '{target_key}'
                
                Dictionaries and other ordered objects must be in the same order for both
                trees. Did you define two dictionaries with the keys in differing orders?
                """
                msg = format_error_message(msg, context_string)
                raise RuntimeError(msg)




def _replicate_leaves(source_treedef: PyTreeDef,
                      target_treedef: PyTreeDef,
                      source_leaves_queue: List[jnp.ndarray],
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
    :param source_leaves_queue: An exit_only queue containing the leaves which may
           need to be duplicated.

           Since the underlying datestructure is such that each leaf in source can
           have a one-to-many map to target leaves, it is the case that once a leaf
           is used it is removed from the queue.
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
    # the same way. We duplicate leaves.
    #
    # When we encounter a tensor leaf on the source tree, we count how many leaves are
    # on the target tree at that same node, then duplicate the source node this many
    # times. This is put into a new list of leafs that is then parsable with the target
    # tree structure in jax.tree_util.tree_unflatten

    # Base case.
    #
    # When the source node is a leaf,
    # we repeat that node a certain number of times and
    # return.

    source_leaves_queue = source_leaves_queue.copy()
    if jax.tree_util.treedef_is_leaf(source_treedef):
        if _is_empty_collection(source_treedef):
            # Bugfix. Collections that are currently empty will be identified
            # as leaves, but not displaced during flatten. We skip those.
            return [], source_leaves_queue

        node_to_repeat = source_leaves_queue.pop(0)
        num_times_to_repeat = _count_linked_leaves(target_treedef)
        updated_source_leaves_section = _repeat_node(node_to_repeat, num_times_to_repeat)
        return updated_source_leaves_section, source_leaves_queue
    elif jax.tree_util.treedef_is_leaf(target_treedef):
        msg = """
        Source tree was not broadcastable.
         
        Source tree had pytree branch node, at position target
        tree had tensor node. This is not allowed.
        
        Did you swap the order you gave your parameters?
        """
        msg = format_error_message(msg, context_message)
        raise RuntimeError(msg)

    # We skipped the base case. That should mean that
    # no leaf was found in source, which means we need to
    # walk the trees in parallel going deeper.


    # We fetch the immediate children from both trees, walk through them
    # recursively, and update our output as we find the expanded
    # leaf sections. We also update the leaves queue in a
    # functional fashion as we go. Once all the leaves have been
    # remapped to be one-to-one, we return the result.

    _validate_children_descent(source_treedef,
                               target_treedef,
                               context_message)

    source_children = jax.tree_util.treedef_children(source_treedef)
    target_children = jax.tree_util.treedef_children(target_treedef)
    output = []
    for i, (source_node, target_node) in enumerate(zip(source_children, target_children)):
        try:

            update, source_leaves_queue = _replicate_leaves(source_node,
                                                           target_node,
                                                           source_leaves_queue,
                                                           context_message)
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
    return output, source_leaves_queue

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

    broadcast_overlap = min(array_a.ndim, array_b.ndim)
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

def broadcast_pytree_shape(source: PyTree,
                           target_structure: PyTree,
                           )->PyTree:
    """
    A function for broadcasting pytrees such
    that their shapes become compatible.

    This operates similar to normal broadcasting, except it is pytree
    speficic and assumes that when a node is a leaf on the source tree,
    and the target tree structure has a branch there, you want to
    fill a structure like the target branch with the source node.

    As a simple example, consider a pair of trees consisting of tree_a, just a tensor,
     and tree_b, a list of tensors. This would be the following

    tree_a = tensor_1a
    tree_b = [tensor_1b,tensor_2b]

    If we perform broadcast_pytree(tree_a, target=tree_b), the result will end
    up being like:

    output = [tensor_1a, target = tensor_1b,
              tensor_1a, target=tensor_2b
             ]

    Notice the PyTree structure at the end now matches tree_b, with the node in that tree
    determining exactly how we are broadcasting!

    :param source: The tree to start from
    :param target_structure: The tree to try to broadcast to
    :return: A broadcast tree. It is guaranteed to have a tree shape that matches one-to-one
             with target if successful. The leaves can be walked cleanly together.
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

    error_context = "An issue occurred while trying to broadcast two pytrees together: "
    source_leaves, source_treedef = jax.tree_util.tree_flatten(source)
    target_leaves, target_treedef = jax.tree_util.tree_flatten(target_structure)

    # Replicates source leaves when one source leaf corresponds to multiple leaves in the
    # target tree def. This ensures the number of leaves is correct.
    final_leaves, remainder = _replicate_leaves(source_treedef, target_treedef, source_leaves, error_context)
    assert len(final_leaves) == len(target_leaves), "error 1: This should never happen. Yell at maintainer"
    assert len(remainder) == 0, "error 2: This should never happen. Yell at maintainer"
    return jax.tree_util.tree_unflatten(target_treedef, final_leaves)


