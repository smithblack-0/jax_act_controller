from typing import Union, Tuple, List, Dict

from jax import numpy as jnp

LeafType = jnp.ndarray
PyTree = Union[LeafType, Tuple['PyTree', ...], List['PyTree'], Dict[str, 'PyTree']]
