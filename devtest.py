import jax
from jax import numpy as jnp
from jax.experimental import checkify
from typing import Callable, Type


class ParameterCapture:
  def __init__(self,
               *args,
               **kwargs
               ):
    self.args = args
    self.kwargs = kwargs
  def __call__(self, function: Callable):
    return function(*self.args, **self.kwargs)


def checkify_and_jit(function: Callable):
  checkified_function = checkify.checkify(function)
  jit_function = jax.jit(checkified_function)


def discharge_side_effects(execution : Tuple[Callable, Capture]):


  errors, output = jit_function()
  return output, errors

def unflatten_and_execute_side_effects(errors, output):
  errors.throw()
  return output
def unflatten_computation(aux_data: BoundCapture, main: Any):
  aux_data.errors.throw()
  return main

  r