import jax
from dataclasses import dataclass
from typing import Dict, Optional, Any
from jax import numpy as jnp
from src.types import PyTree


@dataclass
class ACTStates:
    """
    A collection of ACT state parameters

    ---- fields ---
    iteration: The iteration the process is currently on.
    residuals: Holds accumulated residuals during the act process
    probabilities: Holds cumulative probabilities during the act process
    defaults: Holds the reset defaults during the act process, in case you reset just a channel.
              Generally, set once and never changed.
    accumulators: Holds the accumulated values.
    updates: Holds the accumulator updates while they are getting gathered.

    """
    def replace(self,
                epsilon: Optional[float] = None,
                iterations: Optional[jnp.ndarray] = None,
                residuals: Optional[jnp.ndarray] = None,
                probabilities: Optional[jnp.ndarray] = None,
                accumulators: Optional[Dict[str, PyTree]] = None,
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

        return ACTStates(epsilon,
                         iterations,
                         residuals,
                         probabilities,
                         self.defaults,
                         accumulators,
                         updates)
    epsilon: float
    iterations: jnp.ndarray
    residuals: jnp.ndarray
    probabilities: jnp.ndarray
    defaults: Dict[str, PyTree]
    accumulators: Dict[str, PyTree]
    updates: Dict[str, Optional[PyTree]]


def state_flatten(state: ACTStates) -> Any:

    output = []
    output.append(state.epsilon)
    output.append(state.iterations)
    output.append(state.residuals)
    output.append(state.probabilities)
    output.append(state.defaults)
    output.append(state.accumulators)
    output.append(state.updates)
    return output, None


def state_unflatten(aux_data: Any, flat_state: Any) -> ACTStates:
    return ACTStates(*flat_state)

jax.tree_util.register_pytree_node(ACTStates, state_flatten, state_unflatten)


@dataclass
class ViewerConfig:
    mask: jnp.ndarray

