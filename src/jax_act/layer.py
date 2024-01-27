"""
A place to put layer based tools for
implementing adaptive computation time

Generally, most practical libraries built on
top of jax end up implementing their functionality
in conceptually discrete 'layers'.

Meanwhile, jitting a act loop requires horrendously nonstandard
operations that are completely outside such a paradyn.

This module contains mechanisms to try to provide a jittable
framework that operates more along the lines of the layers
expected by an ACT library
"""
from typing import Tuple, Union, List, Optional, Callable, Dict

import jax.lax

from src.jax_act import utils
from src.jax_act.controller import ACT_Controller
from src.jax_act.builder import ControllerBuilder
from src.jax_act.types import PyTree
from abc import ABC, abstractmethod
from jax import numpy as jnp

class ACT_Layer:

    @staticmethod
    def is_act_not_complete(combined_state: Tuple[ACT_Controller, PyTree])->bool:
        controller, _ = combined_state
        return ~controller.is_completely_halted

    def execute_wrapped_user_layer(self,
                           combined_state: Tuple[ACT_Controller, PyTree],
                           )->Tuple[ACT_Controller, PyTree]:
        return self.layer(*combined_state)

    def __init__(self,
                 make_controller: Callable[[PyTree], ACT_Controller],
                 act_layer: Callable[[ACT_Controller, PyTree], Tuple[ACT_Controller, PyTree]]
                 ):
        self.make_controller = make_controller
        self.layer = act_layer

    def __call__(self,
                 initial_state: PyTree
                 )->Dict[str, PyTree]:

        error_context_message = "An error occurred while performing adaptive computation time"
        controller = self.make_controller(PyTree)
        if not isinstance(controller, ACT_Controller):
            msg = f"""
            A ACTController was not successfully made. 
            The return from provided user function 'make_controller' did not
            return a controller object. This is not allowed. 
            """
            msg = utils.format_error_message(msg, error_context_message)
            raise TypeError(msg)

        combined_state = (controller, initial_state)
        combined_state = jax.lax.while_loop(cond_fun=self.is_act_not_complete,
                                            body_fun=self.execute_wrapped_user_layer,
                                            init_val=combined_state
                                            )
        controller, output_state = combined_state
        return controller.accumulators,



class ACTMixin:
    """
    A mixin designed to add supporting mechanism for
    adaptive computation time
    """
    def get_builder(self,
                    batch_shape: Union[int, List[int]],
                    dtype: Optional[jnp.dtype] = None,
                    epsilon: float = 1e-4
                    )->ControllerBuilder:
        """
        Returns a new builder. Use it in method make_controller

        :param batch_shape: The shape of the batches you will be handling
        :param dtype: The main dtype of your tensors
        :param epsilon: The act epsilon. You can usually leave it alone
        :return: A ControllerBuilder instance.
        """
        return ControllerBuilder.new_builder(batch_shape,
                                             dtype,
                                             epsilon)

    @abstractmethod
    def make_controller(self)->ACT_Controller:
        """
        An abstract method that must be implemented. This
        should be able to make
        :return:
        """
        pass
    @abstractmethod
    def run_act_layer(self,
                       controller: ACT_Controller,
                       state: PyTree
                       )->Tuple[ACT_Controller, PyTree]:
        """
        An abstract method, it should accept
        an internal state of some sort, process it through
        the layer, then return the resulting controller
        and state.

        :param state:
        :return:
        """
        pass

    def __call__(self,
                 initial_states: PyTree
                 )->Tuple[ACT_Controller, PyTree]:


