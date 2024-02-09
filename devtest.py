import jax
from jax import numpy as jnp
from jax.experimental import checkify
from typing import Callable, Type

def print_branch():
    message = "This has been printed"
    jax.debug.print(message)

def pass_branch():
    pass


@jax.jit
def test_printing(value: float):
    flag = value > 0
    jax.lax.cond(flag, print_branch, pass_branch)

test_printing(-0.2)
test_printing(3.0)

