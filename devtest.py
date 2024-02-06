import jax
from jax import numpy as jnp
from jax.experimental import checkify
def discharge_throw_effects(function: Callable):
  checkified_function = checkify.checkify(function)
  jit_checkified_function = jax.jit(checkified_function)
  def wrapper(*args, **kwargs):
    errors, output = jit_checkified_function(*args, **kwargs)
    return output
  return jax.tree_util.Partial(wrapper)

def restore_throw_effects(function: Callable):
  checkified_function = checkify.checkify(function)
  jit_checkified_function = jax.jit(checkified_function)
  def wrapper(*args, **kwargs):
    errors, output = jit_checkified_function(*args, **kwargs)
    errors.throw()
    return output
  return wrapper

def checkify_and_jit_compile_function(function: Callable):
  jax.tree_util.register_pytree_node(function,
                                     discharge_throw_effects,
                                     restore_throw_effects)
  return function

@checkify_and_jit_compile_function
def test_will_throw():
  checkify.check(False, "This should throw")

jit_function = jax.jit(test_will_throw)

jit_function()
