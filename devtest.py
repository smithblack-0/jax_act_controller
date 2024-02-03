import jax
from jax import numpy as jnp

test = [jnp.ones([2, 3]), {"item" :jnp.ones([5, 6, 7])}]
print(test[1, (-0):])