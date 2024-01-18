import jax
from jax import numpy as jnp
from typing import Callable
from src.types import PyTree

def merge_pytrees(function: Callable[[jnp.ndarray,
                                       jnp.ndarray],
                                      jnp.ndarray],
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
