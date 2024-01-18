from src.immutable import Immutable
from src.states import ACTStates, ViewerConfig
from src import utils
from jax import numpy as jnp

class ACT_Viewer(Immutable):

    def _apply_mask(self,
                    mask: jnp.ndarray,
                    tensor: jnp.ndarray):
        """
        Applies the given mask to the given tensor by means of
        multiplication. Broadcasts to fit

        :return: The tensor, after the mask is applied
        """

        mask = utils.setup_left_broadcast(mask, tensor)
        return mask*tensor

    @property
    def probabilities(self):
        return utils.apply_mask(self.configuration.mask, self.states.probabilities)



        ## Display functions
        #
        # These are used primarily to make it easy to look at
        # and use your gathered data and statistics.
        def mask_data(self,
                      mask: jnp.ndarray,
                      lock: bool = True,
                      ) -> 'ACT_Controller':
            """
            A helper function for creating controllers with masked
            data.

            A mask must be of the same shape as the batch shape,
            or the probability shape. So long as it is, it will
            be applied to each and every tensor inside the controller,
            then a new controller with the new tensors will be
            returned.

            This can be useful for isolating particular features for examination.
            However, by default, it results in the locking of the controller. This
            can be overridden, but exists to prevent a user from trying to use the
            returned controller to continue the act process.

            :param mask: The mask to apply. Must have batch shape
            :param lock: Whether to lock the new controller.
            :return: A new ACT_Controller instance where the mask has been applied.
            :raise: ValueError, if the mask and batch shape are not the same
            :raise: ValueError, if you attempt to mask before committing all updates.
            """

            if mask.shape != self.probabilities.shape:
                raise ValueError(f"Mask of shape {mask.shape} does not match batch shape of {self.probabilities.shape}")
            for item in self.state.updates:
                if item is not None:
                    raise ValueError(f"Data mask was attempted while updates were not committed.")

            iterations = self.iterations * mask
            residuals = self.residuals * mask
            probabilities = self.probabilities * mask

            # TODO: Fix

            update_func = lambda tree: self._apply_mask(mask, tree)
            accumulators = {name: jax.tree_util.tree_map(update_func, value)
                            for name, value
                            in self.accumulators.items()}

            state = self.state.replace(iterations=iterations,
                                       residuals=residuals,
                                       probabilities=probabilities,
                                       accumulators=accumulators
                                       )

            return ACT_Controller(state)

        def mask_unhalted_data(self) -> 'ACT_Controller':
            """
            Creates a new controller, in which all tensors which
            are corrolated with unhalted data are masked by filling with
            zeros.

            :return: A ACT_Controller instance, where the tensors are masked. To
                    prevent inadvertant usage, the controller is blocked from
                    additional act action.

            :raise: ValueError, if you attempt to mask before commit ing all updates
            """

            unhalted_selector = ~self.halted_batches
            return self.mask_data(unhalted_selector)

        def mask_halted_data(self) -> 'ACT_Controller':
            """
            Creates a new controller, in which all tensors which
            are corrolated with halted data are masked by filling with
            zeros.

            :return: A ACT_Controller instance, where the tensors are masked. To
                    prevent inadvertant usage, the controller is blocked from
                    additional act action.

            :raise: ValueError, if you attempt to mask before commit ing all updates
            """

            halted_selector = self.halted_batches
            return self.mask_data(halted_selector)

    def __init__(self,
                 state: ACTStates,
                 configuration: ViewerConfig):
        super().__init__()
        self.states = state
        self.make_immutable()



