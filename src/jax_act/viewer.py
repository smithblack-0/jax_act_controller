import jax
from jax import numpy as jnp

from typing import Dict, Callable, Optional, Tuple, Any

from src.jax_act.types import PyTree
from src.jax_act.immutable import Immutable
from src.jax_act.states import ACTStates
from src.jax_act import utils


class ACTViewer(Immutable):
    """
    A class used to isolate views of act features.

    It is an immutable class that can be made from
    a controller class. It functions by allowing
    the user to provide "masks" that will result
    in sections becoming zero.

    The mask can be user defined, but must be of
    shape batch_shape.

    The class is immutable - any attempt to modify
    it will just return a new instance

    ----- properties
    probabilities: The masked cumulative probabilities
    residuals: The masked residuals
    iterations: The masked iterations
    accumulators: The masked accumulators
    defaults: Default values
    updates: Update values.

    ---- methods ---

    mask_data: Accepts a user-defined mask of shape probabilities.
               Applies it to the internal probabilities.
    transform: A more general transform operator. This will hunt down each tensor, then call
               the provided function. A new viewer instance is returned.
    unhalted_only: Applies a new viewer instance in which all finished (halted) tensor elements
                      have been filled with zero
    halted_only: Returns a new viewer instance in which all in-progress tensor elements
                   (unhalted) have been filled with zero.
    save: Returns the internal state. In theory, any act class with a load function could then
          load it. In practice, make sure you actually want to do that
    load: Load a state.
    """

    # Defined identically to controller: Properties.
    @property
    def probabilities(self)->jnp.ndarray:
        return self.state.probabilities
    @property
    def residuals(self)->jnp.ndarray:
        return self.state.residuals
    @property
    def iterations(self)->jnp.ndarray:
        return self.state.iterations
    @property
    def accumulators(self)->Dict[str, PyTree]:
        return self.state.accumulators

    @property
    def defaults(self)->Dict[str, PyTree]:
        return self.state.defaults
    @property
    def updates(self)->Dict[str, Optional[PyTree]]:
        return self.state.updates
    @property
    def halt_threshold(self)->float:
        return 1-self.state.epsilon
    @property
    def halted_batches(self)->jnp.ndarray:
        return self.probabilities > self.halt_threshold

    #
    def transform(self,
                  function: Callable[[jnp.ndarray], jnp.ndarray]
                  )->'ACTViewer':
        """
        A generalized function for transforming the internal state
        functions in a more generalized way. It is generally expected
        to be used to implement reshape or mask logic.

        The provided callable should be able to accept any tensor
        of batch_shape, and will produce something else in return

        IMPORTANT: If you are trying to pass a function in a jit context,
        you may need to wrap it in partial under jax.tree_utils.Partial.

        :param function: The function to apply
        :return: A new ACTViewer with the results of the transform
        """
        try:
            iterations = function(self.iterations)
            residuals = function(self.residuals)
            probabilities = function(self.probabilities)

            accumulators = {name: jax.tree_util.tree_map(function, value)
                            for name, value
                            in self.accumulators.items()}
            defaults = {name: jax.tree_util.tree_map(function, value)
                        for name, value
                        in self.defaults.items()}
            def update_applier(value):
                if value is not None:
                    return function(value)
                return None

            updates = {name: jax.tree_util.tree_map(update_applier, value)
                        for name, value
                        in self.defaults.items()}

        except Exception as err:
            msg = "An issue occurred in using the provided function"
            raise RuntimeError(msg) from err

        state = self.state.replace(iterations=iterations,
                                   residuals=residuals,
                                   probabilities=probabilities,
                                   accumulators=accumulators,
                                   defaults=defaults,
                                   updates=updates
                                   )
        return ACTViewer(state)

    def mask_data(self,
              mask: jnp.ndarray,
              ) -> 'ACTViewer':
        """
        A function that will create a new viewer where the elements
        left false in the provided mask are filled with zeros.

        A mask must be of the same shape as the batch shape. It
        will then be applied to every contained tensor, making elements
        that do not satisfy the mask zero.

        :param mask: The mask to apply. Must have batch shape. True means keep.
        :return: A new ACT_Controller instance where the mask has been applied.
        :raise: ValueError, if the mask and batch shape are not the same
        :raise: ValueError, if you attempt to mask before committing all updates.
        """
        if mask.dtype != jnp.bool_:
            raise ValueError(f"Mask was not made up of bool dtypes.")

        if mask.shape != self.probabilities.shape:
            raise ValueError(f"Mask of shape {mask.shape} does not match batch shape of {self.probabilities.shape}")

        def update_function(tensor: jnp.ndarray)->jnp.ndarray:
            broadcastable_mask = utils.setup_left_broadcast(mask, tensor)
            return broadcastable_mask*tensor

        return self.transform(update_function)

    def unhalted_only(self) -> 'ACTViewer':
        """
        Creates a new viewer, in which all tensors that had
        reached the halted state are masked away. As such, you
        only see the in progress entries.

        :return: A ACT_Controller instance, where the tensors are masked. To
                prevent inadvertant usage, the controller is blocked from
                additional act action.
        """

        unhalted_selector = ~self.halted_batches
        return self.mask_data(unhalted_selector)

    def halted_only(self) -> 'ACTViewer':
        """
        Creates a new viewer instance, in which all tensors which
        are corrolated with halted data are masked by filling with
        zeros. What is left is only unhalted information.

        The displayed tensors then become suitable for accumulator in
        external processes.

        :return: A ACT_Viewer instance, where the tensors are masked. To
                prevent inadvertant usage, the controller is blocked from
                additional act action.
        """

        halted_selector = self.halted_batches
        return self.mask_data(halted_selector)

    def save(self)->ACTStates:
        """
        Returns a save of the class state.
        :return: State
        """
        return self.state

    @classmethod
    def load(cls, state: ACTStates)->'ACTViewer':
        """
        Loads into the class a state
        :param state: The state to load
        :return: A ACTViewer
        """
        return cls(state)
    def __init__(self, state: ACTStates):
        """
        Generally, as a user, you will never use this

        Instead, create a viewer from a controller.
        :param state: The state to lad.
        """
        super().__init__()
        self.state = state
        self.make_immutable()

def flatten_viewer(viewer: ACTViewer)->Tuple[Any, Any]:
    state = viewer.save()
    flat_state, tree_def = jax.tree_util.tree_flatten(state)
    return flat_state, tree_def

def unflatten_viewer(tree_def: Any, flat_state: Any)->ACTViewer:
    state = jax.tree_util.tree_unflatten(tree_def, flat_state)
    return ACTViewer.load(state)

jax.tree_util.register_pytree_node(ACTViewer, flatten_viewer, unflatten_viewer)

