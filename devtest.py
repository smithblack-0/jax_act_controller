import jax
from jax import numpy as jnp
from jax.experimental import checkify

@checkify.checkify
def to_check(array: jnp.ndarray):
    probs = array > 1.0
    checkify.check(~jnp.any(probs), "Test")

err, output = to_check(jnp.array([0, 2]))
print(err)
err, output = to_check(jnp.array([0, 0.1, 0.2]))
print(err)