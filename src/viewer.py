import jax
from jax import numpy as jnp

from typing import Dict

from src.types import PyTree
from src.immutable import Immutable
from src.states import ACTStates, ViewerConfig
from src import utils


class ACT_Viewer(Immutable):
    """
    A class used to isolate views of act features.

    It is primarily useful for extracting information associated
    only with particular dimensions.
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
    def halt_threshold(self)->float:
        return 1-self.state.epsilon
    @property
    def halted_batches(self)->jnp.ndarray:
        return self.probabilities > self.halt_threshold

    # View functions and functionality
    def mask_data(self,
              mask: jnp.ndarray,
              ) -> 'ACT_Viewer':
        """
        A function that will create a new viewer with data that has been masked.

        A mask must be of the same shape as the batch shape,
        or the probability shape. So long as it is, it will
        be applied to each and every tensor inside the controller,
        then a new controller with the new tensors will be
        returned.

        This can be useful for isolating particular features for examination.

        :param mask: The mask to apply. Must have batch shape
        :param lock: Whether to lock the new controller.
        :return: A new ACT_Controller instance where the mask has been applied.
        :raise: ValueError, if the mask and batch shape are not the same
        :raise: ValueError, if you attempt to mask before committing all updates.
        """
        if mask.dtype != jnp.bool_:
            raise ValueError(f"Mask was not made up of bool dtypes.")

        if mask.shape != self.probabilities.shape:
            raise ValueError(f"Mask of shape {mask.shape} does not match batch shape of {self.probabilities.shape}")

        iterations = self.iterations * mask
        residuals = self.residuals * mask
        probabilities = self.probabilities * mask

        def update_func(leaf: jnp.ndarray):
            broadcastable_mask = utils.setup_left_broadcast(mask, leaf)
            return broadcastable_mask*leaf
        accumulators = {name: jax.tree_util.tree_map(update_func, value)
                        for name, value
                        in self.accumulators.items()}

        state = self.state.replace(iterations=iterations,
                                   residuals=residuals,
                                   probabilities=probabilities,
                                   accumulators=accumulators
                                   )

        return ACT_Viewer(state)

    def in_progress_only(self) -> 'ACT_Viewer':
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

    def results_only(self) -> 'ACT_Viewer':
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
    def __init__(self, state: ACTStates):
        super().__init__()
        self.state = state
        self.make_immutable()



