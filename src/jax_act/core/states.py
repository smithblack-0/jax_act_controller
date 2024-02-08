"""
INTERNAL: Code in this section is entirely internal,
and subject to change on a whim.
"""
import textwrap

import jax
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, Tuple, Callable, List
from jax import numpy as jnp
from jax.experimental import checkify
from src.jax_act.core.types import PyTree
from src.jax_act.core import utils

@dataclass
class EditorConfig:
    """
    A boolean collection of parameters
    designed to indicate whether a field
    should be getting edited.
    """
    edit_epsilon: bool
    edit_iterations: bool
    edit_residuals: bool
    edit_probabilities: bool
    edit_accumulator: bool
    edit_defaults: bool
    edit_updates: bool
    edit_depression_constant: bool
@dataclass
class ACTStates:
    """
    A collection of ACT state parameters and validation logic

    Validation logic that is relevant to the state parameters
    are defined at the location of their relationship. However,
    this logic is not actually implemented, and left to
    the upstream user to use.

    ---- fields ---
    epsilon: A float, used to calculate a halting threshold. Should be a probability
    iterations: A batch shaped tensor. It is actually of floating dtype.
    residuals: Holds accumulated residuals during the act process.
    probabilities: Holds cumulative probabilities during the act process
    defaults: Holds the reset defaults during the act process, in case you reset just a channel.
    accumulators: Holds the accumulated values.
    updates: Holds the accumulator updates while they are getting gathered.
    depression_constant: What to multiply the halting probabilities by
    error_mode: Which error mode to operate under.
    """
    def execute_validation(self,
                           operand: Any,
                           predicates: List[Tuple[Callable[[Any], bool], str]],
                           context_message: str
                           )->Any:
        """
        Executes a gathered validation group in one of the
        three validation modes.

        :param operand: What needs to be validated
        :param predicates: A list of predicate tests, and messages to return when failed
            - First item will take in an operand, and return a bool if failed
            - Second item is a string containing a message to read when failed
        """
        if self.error_mode == ErrorModes.checkify:
            for predicate, msg in predicates:
                msg = utils.format_error_message(msg, context_message)
                checkify.check(predicate(operand), msg)
        elif self.error_mode == ErrorModes.debug:
            for predicate, msg in predicates:
                msg = utils.format_error_message(msg, context_message)
                if not predicate(operand):
                    raise ValueError(msg)
        return operand
    def validate_epsilon(self,
                         epsilon: float,
                         context_msg: str)->float:
        predicates = [
            (lambda epsilon: epsilon >= 0.0,
             "Epsilon was not found to be greater than zero. This is not allowed"),
            (lambda epsilon: epsilon <= 1.0,
             "Epsilon was not found to be less than or equal to 1.0. This is not allowed"
             )
        ]
        return self.execute_validation(epsilon, predicates, context_msg)

    def validate_iterations(self,
                            iterations: jnp.ndarray,
                            context_msg: str
                            )->jnp.ndarray:
        predicates = [
            # Handle floating point issues
            (lambda iterations : jnp.issubdtype(iterations, jnp.floating),
            f"""An issue occurred while validating iterations
            
            For reasons of interpolation, the iterations tensor must be a floating
            point tensor. It was found instead to be of type {iterations.dtype}
            """
            ),
            # Handle iterations is greater than or equal to one
            (lambda iterations : jnp.all(iterations >= 0),
             """An issue occurred while validating iterations
             
             Iterations had elements that were less than 0. This
             is not allowed, as iterations is a counter.
             """)
        ]
        return self.execute_validation(iterations, predicates, context_msg)

    def validate_residuals(self, residuals: jnp.ndarray, context_msg):
        predicates = [
            (lambda residuals : jnp.all(residuals >=0.0),
             """
             An issue occurred while validating residuals. 
             
             Residuals are probabilities, and should be greater than
             or equal to zero. However, some residuals had values
             less than zero
             """
             ),



        ]
    def replace(self,
                epsilon: Optional[float] = None,
                iterations: Optional[jnp.ndarray] = None,
                residuals: Optional[jnp.ndarray] = None,
                probabilities: Optional[jnp.ndarray] = None,
                accumulators: Optional[Dict[str, PyTree]] = None,
                defaults: Optional[Dict[str, PyTree]] = None,
                updates: Optional[Dict[str, Optional[PyTree]]] = None,
                depression_constant: Optional[float] = None,
                error_mode: Optional[str] = None
                )->'ACTStates':

        epsilon = self.epsilon if epsilon is None else epsilon
        iterations = self.iterations if iterations is None else iterations
        residuals = self.residuals if residuals is None else residuals
        probabilities = self.probabilities if probabilities is None else probabilities
        accumulators = self.accumulators if accumulators is None else accumulators
        updates = self.updates if updates is None else updates
        defaults = self.defaults if defaults is None else defaults

        if depression_constant is None:
            depression_constant = self.depression_constant
        else:
            depression_constant = self.validate_depression_constant(depression_constant)
        if error_mode is None:
            error_mode = self.error_mode
        else:
            error_mode = self.validate_error_mode(error_mode)


        return ACTStates(epsilon,
                         iterations,
                         residuals,
                         probabilities,
                         defaults,
                         accumulators,
                         updates,
                         depression_constant,
                         error_mode
                         )
    epsilon: float
    iterations: jnp.ndarray
    residuals: jnp.ndarray
    probabilities: jnp.ndarray
    defaults: Dict[str, PyTree]
    accumulators: Dict[str, PyTree]
    updates: Dict[str, Optional[PyTree]]
    depression_constant: float
    error_mode: str

# We define the functions to represent states as a pytree,
# and then register the tree.
#
# This involves manually pulling out and then flattening
# the keys from the state, and doing the reverse to put
# everything back together.
def state_flatten(state: ACTStates) -> Tuple[Any, Any]:
    state = asdict(state)

    aux_out = []
    values = []
    for key, value in state.items():
        value, aux = jax.tree_util.tree_flatten(value)

        aux_out.append(aux)
        values.append(value)
    return zip(state.keys(), values), aux_out

def state_unflatten(aux_data: Any, flat_state: Any) -> ACTStates:

    values = []
    for value, tree_def in zip(flat_state, aux_data):
        value = jax.tree_util.tree_unflatten(tree_def, value)
        values.append(value)
    return ACTStates(*values)

jax.tree_util.register_pytree_with_keys(ACTStates, state_flatten, state_unflatten)