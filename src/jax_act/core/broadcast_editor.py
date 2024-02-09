"""

A manipulator designed to allow the easy manipulation of
underlying state in terms of probabilities broadcast across
internal tensors.


Mathematical Ideal:
    We define a new operand object, called a BroadcastEditor, which has the following fields.

    1) A field called states containing an ACTStates. This will be updated
    2) A field called config containing a sequence of boolean entries, one per element
       of states.

    An underlying act feature is said to be "on" or being "edited" when
    it's corresponding configuration action is on. Operators are updated
    with regards to this operand object to function as follows:

    - An operation between a tensor and a BroadcastEditor is equal to distributing
      the operation onto all active config states, executing against all tensor collections,
      then rebuilding the broadcast editor
    - An operation between a tensor collection, or "PyTree", is equal to attempting
      to broadcast the pytree between all active config states.
    - An operation between a BroadcastEditor and a BroadcastEditor is only possible
      so long as the configs are identical and the inactive states are identical.
    - All operations return a new instance of BroadcastEditor with the updates applied.

    Additionally, due to limitations of jax, you cannot
    expect the class to yell
"""

import jax
from jax import numpy as jnp
from src.jax_act.core.types import PyTree
from src.jax_act.core.controller import ACT_Controller
from src.jax_act.core.states import ACTStates, EditorConfig
from src.jax_act.core import utils
from typing import Callable, Union, Tuple, Any
from dataclasses import dataclass
PyTree = Union[PyTree, ACTStates]


class BroadcastEditor:
    # TODO:
    #    - Properties passthroughs
    #    - Methods:
    #        - Main binary operator
    #        - Main unitary operator
    #        - Various arithmetic methods
    #        - Various magic methods
    #    - Docstring

    def __init__(self,
                 state: ACTStates,
                 config: EditorConfig):
        self.state = state
        self.config = config
