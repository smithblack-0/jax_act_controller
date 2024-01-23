import jax
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, Tuple
from jax import numpy as jnp
from src.jax_act.types import PyTree


@dataclass
class ACTStates:
    """
    A collection of ACT state parameters and other
    stateful items needed to perform adaptive computation time.

    ---- fields ---
    epsilon: A float, used to calculate a halting threshold.

    iterations: A batch shaped tensor, counting how many iterations have been done per batch
    residuals: Holds accumulated residuals during the act process.
    probabilities: Holds cumulative probabilities during the act process
    defaults: Holds the reset defaults during the act process, in case you reset just a channel.
    accumulators: Holds the accumulated values.
    updates: Holds the accumulator updates while they are getting gathered.
    """
    def replace(self,
                epsilon: Optional[float] = None,
                iterations: Optional[jnp.ndarray] = None,
                residuals: Optional[jnp.ndarray] = None,
                probabilities: Optional[jnp.ndarray] = None,
                accumulators: Optional[Dict[str, PyTree]] = None,
                defaults: Optional[Dict[str, PyTree]] = None,
                updates: Optional[Dict[str, Optional[PyTree]]] = None,
                )->'ACTStates':
        """
        Replace one or more feature, while leaving everything else
        the exact same.
        :return: A new ACT states
        """
        if epsilon is None:
            epsilon = self.epsilon

        if iterations is None:
            iterations = self.iterations
        if residuals is None:
            residuals = self.residuals
        if probabilities is None:
            probabilities = self.probabilities
        if accumulators is None:
            accumulators = self.accumulators
        if updates is None:
            updates = self.updates
        if defaults is None:
            defaults = self.defaults

        return ACTStates(epsilon,
                         iterations,
                         residuals,
                         probabilities,
                         defaults,
                         accumulators,
                         updates)
    epsilon: float
    iterations: jnp.ndarray
    residuals: jnp.ndarray
    probabilities: jnp.ndarray
    defaults: Dict[str, PyTree]
    accumulators: Dict[str, PyTree]
    updates: Dict[str, Optional[PyTree]]


# We define the functions to represent ACTStates as a pytree,
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

