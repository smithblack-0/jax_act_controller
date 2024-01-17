import jax
from typing import Callable
from src.types import PyTree

def tree_multimap(func: Callable,
                   *trees: PyTree
                   ) -> PyTree:
    """
    Jax, strangly, has no tree multimap function. This means
    walking through trees with the same structure, to like
    leaves, in parallel, is not immediately possible. 

    This function provides that behavior. 
    :param func: The function to map
    :param trees: The trees to map on.
    :return: The reassembled PyTree
    """

    # Merging PyTrees turns out to be more difficult than
    # expected.
    #
    # Jax, strangly, has no multitree map, which means walking through the
    # nodes in parallel across multiple trees is not possible. Instead, we flatten
    # both trees deterministically, walk through the leaves, and then restore the
    # original structure.

    assert len(trees) > 0
    treedef = jax.tree_structure(trees[0])
    
    flat_trees = [jax.tree_flatten(item) for item in trees]
    