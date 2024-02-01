import flax
import flax.linen as nn
import jax

from jax import numpy as jnp
from jax import random
import optax
import unittest
from flax.training import train_state
from typing import Tuple
from src.jax_act import ACT_Controller, ControllerBuilder, execute_act


class Bias(nn.Module):
    """
    A simple layer that exists to verify
    gradient descent is possible.

    This layer will take the incoming tensor,
    add parameters to it, then sum the result
    together.
    """
    num_parameters: int
    def setup(self):
        self.bias = self.param(
            "parameters_to_sum",
            nn.initializers.uniform(),
            (self.num_parameters, ) ,
            jnp.float32
        )
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        biased = x + self.bias
        return biased



class ACT_Executer_Layer(nn.Module):
    num_parameters: int
    def setup(self):
        self.act_helper = ACTHelperLayer(self.num_parameters)

    def __call__(self, x: jnp.ndarray):
        return execute_act(self.act_helper,
                           x)




key1, key2 = random.split(random.key(0))
x = random.normal(key1, [10])
layer = Bias()
params = layer.init(key2, x)
output = layer.apply(params, x)