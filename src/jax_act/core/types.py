"""


Definition of what a PyTree is

A PyTree is a nested python structure of
Tuples, Lists, and Dicts with ndarrays as leaves.

The leaves should be floating tensors.

STABLE: Code in this section will never be modified in a way that breaks backwards
compatibility
"""



from typing import Union, Tuple, List, Dict, Protocol
from jax import numpy as jnp

LeafType = jnp.ndarray
PyTree = Union[LeafType, Tuple['PyTree', ...], List['PyTree'], Dict[str, 'PyTree']]

LeafType = List[int]
PyTreeShapes = Union[LeafType, Tuple['PyTree', ...], List['PyTree'], Dict[str, 'PyTree']]
