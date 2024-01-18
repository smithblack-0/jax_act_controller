"""
Creation contains the builders and other mechanisms
needed to create an ACT instance. It also includes
some degree of error checking.
"""

from typing import Optional, Any, Tuple, Callable
from src.states import ACTStates
from src.immutable import Immutable
import jax.tree_util
import textwrap
from jax import numpy as jnp
from src.types import PyTree


class ACT_Controller(Immutable):
    """
    A controller for the ACT process.

    This is an ACT controller designed to be
    setup in the associated builder, and which will accumulate
    information across act steps for all targeted quantities.

    The class must be initialized with a ACTStates object, and
    will generally return a new instance of itself whenever it
    is modified.

    The controller is completely immutable. Once initialized, if a method
    producing a state change is called, a new controller is returned instead.

    The class is designed to be provided wholesale as a return from a
    builder. It would be very rare for a user to directly configure the state.
    

    ----- Direct Properties

    probabilities: The cumulative probabilities for each batch element
    residuals: The residuals on each batch element
    iterations: The iteration each batch is on, or alternatively where
    accumulators: The accumulated results, so far.

    ----- Properties ----

    halt_threshold: Above this cumulative probability a accumulator is considered halted
    halted_batches: A bool tensor of batches that are halted. True means halted.
    is_completely_halted: If every batch element is halted, this is true
    is_any_halted: If at least one batch element is halted, this is true.

    ----- Main Methods -----

    Main methods are designed to be used as part of your act
    computation process.

    cache_update: A function that accepts a PyTree and should be called  once per
                  act operation for each accumulator. Updates should be the same
                  shape as the originally defined accumulator.

                  It creates a new instance that stores the update to be integrated into
                  the accumulator. These updates will be used once iterate_act is called.

    iterate_act: A function that is called once updates are waiting in all accumulators, at the
                 end of the act process. It is called with the halting probabilities. It uses the
                 probabilities to update the accumulators and the cumulative probabilities.

                 It also updates numerous other features which are updated, such as iteration and
                 residuals.
    reset: A function that places halted act channels back into their default
           condition. It will return a new controller instance.

    ---- Display Methods -----


    ---- Usage ----

    To use the controller, you will first need to setup the accumulation
    channels using a factory method, the provided builder, or by hand. Then,
    you use

    ----- Locking ----

    One thing worth mentioning is that certain actions will corrupt the
    tensors in the controller for data analysis purposes. These will result
    in a returned controller in the 'locked' state. This means that functions,
    like cache_update and iterate_act, that are used for actual act work stop
    working and throw if called.

    If you are having this issue, the class just saved you several hours of
    debugging. You need to consume the class in view mode immediately, then
    continue using the instance from beforehand.

    For example, the following code will cause issues

    ```
    statistics = []
    for ...:
        ... your code
        contr

        if CONDITION:
            controller = controller.mask_unhalted_data:


    ```


    """


    # Direct properties

    @property
    def is_locked(self)->bool:
        return self.state.is_locked
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
    def accumulators(self)->PyTree:
        return self.state.accumulators

    # Inferred properties
    @property
    def halt_threshold(self)->float:
        return 1-self.state.epsilon
    @property
    def halted_batches(self)->jnp.ndarray:
        return self.probabilities > self.halt_threshold

    @property
    def is_completely_halted(self)->bool:
        return jnp.all(self.halted_batches)

    @property
    def is_any_halted(self)->bool:
        return jnp.any(self.halted_batches)

    # Helper logic
    @staticmethod
    def _setup_left_broadcast(tensor: jnp.ndarray,
                              target: jnp.ndarray
                              ) -> jnp.ndarray:
        """
        Sets up tensor by unsqueezing dimensions for
        a left broadcast with the target.

        Returns the unsqueezed tensor.

        :param tensor: The tensor to expand
        :param target: The target whose length to match
        :return: The unsqueezed tensor
        """

        assert len(tensor.shape) <= len(target.shape)
        while len(tensor.shape) < len(target.shape):
            tensor = tensor[..., None]
        return tensor
    @staticmethod
    def _merge_pytrees(function: Callable[[jnp.ndarray,
                                          jnp.ndarray],
                                         jnp.ndarray],
                      primary_tree: PyTree,
                      auxilary_tree: PyTree,
                      )->PyTree:
        """
        Used to merge two pytrees together.

        This deterministically walks the primary and auxilary
        trees, then calls function when like leaves are reached. The
        results are created into a new tree.

        :param function: A function that accepts first a primary leaf, then a auxilary leaf
        :param primary_tree: The primary tree to draw from
        :param auxilary_tree: The auxilary tree to draw from
        :return: A new PyTree
        """
        # Merging PyTrees turns out to be more difficult than
        # expected.
        #
        # Jax, strangly, has no multitree map, which means walking through the
        # nodes in parallel across multiple trees is not possible. Instead, we flatten
        # both trees deterministically, walk through the leaves, and then restore the
        # original structure.

        treedef = jax.tree_util.tree_structure(primary_tree)
        primary_leaves = jax.tree_util.tree_flatten(primary_tree)[0]
        auxilary_leaves = jax.tree_util.tree_flatten(auxilary_tree)[0]

        assert len(primary_leaves) == len(auxilary_leaves)  # Just a quick sanity check.

        new_leaves = []
        for primary_leaf, aux_leaf in zip(primary_leaves, auxilary_leaves):
            update = function(primary_leaf, aux_leaf)
            new_leaves.append(update)

        return jax.tree_util.tree_unflatten(treedef, new_leaves)
    def _check_if_locked(self):
        if self.locked:
            msg = f"""
            An attempt was made to run ACT with a locked controller. 

            Calling any function other than cache_updates, reset_batches, or iterate_act will
            immediately result in the returned controller being set into the locked
            state. This prevents you from trying to continue with ACT on a controller
            set in an information display mode.

            If you need to fetch information, and you need to continue act, then
            just make a new controller with a different name, and keep the original controller
            around
            """
            msg = textwrap.dedent(msg)
            raise RuntimeError(msg)



    def _process_probabilities(self,
                               halting_probabilities: jnp.ndarray
                               )->Tuple[jnp.ndarray,
                                        jnp.ndarray,
                                        jnp.ndarray
                                        ]:
        """
        Process the halting probabilities into the clamped
        halting probabilities, the residuals, and the new cumulative probabilities
        :param halting_probabilities: The current halting probabilities
        :return: The clamped halting probabilities
        :return: The updated residuals
        :return: The updated cumulative probabilities
        """

        # We need to know what batches are just now halting
        # to capture that information in the residuals, and
        # what tensors will be halting to properly trim
        # the halting probabilities. Computations assume that
        # something that is going to be halted, but are not
        # currently halted, will halt in this iteration.

        will_be_halted = (halting_probabilities + self.probabilities) > self.halt_threshold
        newly_halting = will_be_halted & ~self.halted_batches

        # Master residuals are updated only the round that we
        # are newly halted. However, the unfiltered residuals
        # are still useful for probability clamping

        raw_residuals = 1 - self.probabilities
        residuals = jnp.where(newly_halting, raw_residuals, self.residuals)

        # Halting probabilities are clamped to the raw residual entries
        # where they will be halted. This ensures total probability can
        # never exceed one.

        halting_probabilities = jnp.where(will_be_halted, raw_residuals, halting_probabilities)

        # Finally, cumulative probabilities are updated to contain the sum
        # of the halting probabilities and the current cumulative probabilities

        probabilities = self.probabilities + halting_probabilities

        return halting_probabilities, residuals, probabilities

    def _update_accumulator(self,
                            accumulator_value: jnp.ndarray,
                            update_value: Optional[jnp.ndarray],
                            halting_probabilities: jnp.ndarray
                            ) -> jnp.ndarray:
        """
        Performs an update step using an accumulator leaf and an
        update leaf, processing the halting probabilities
        to match.

        :param accumulator_value: The current accumulator
        :param update_value: The value to include in the update
        :param halting_probabilities: The halting probabilities
        :return: The new accumulator.
        """

        if update_value is None:
            msg = f"""
            There was no cached update stored for this act iteration. This
            makes forming an update impossible.
            """
            msg = textwrap.dedent(msg)
            raise RuntimeError(msg)

        if accumulator_value.shape != update_value.shape:
            msg = f"""
            Original accumulator leaf of shape {accumulator_value.shape} does not match
            update accumulator leaf of shape {update_value.shape}. This is not allowed
            """
            msg = textwrap.dedent(msg)
            raise RuntimeError(msg)

        # We will need the halting probabilities and halting batches tensors
        # to be broadcastable, from the left, with the accumulator shapes.
        #
        # We will often need to unsqueeze on the last dimension to make this
        # happen. This section accomplishes this.
        halted_batches = self.halted_batches
        halting_probabilities = self._setup_left_broadcast(halting_probabilities, accumulator_value)
        halted_batches = self._setup_left_broadcast(halted_batches, accumulator_value)

        # We weight then add. The result is then created into an update, but
        # only accumulators which have not already reached a halting state get updated.

        new_accumulators = accumulator_value + halting_probabilities * update_value
        return jnp.where(halted_batches, accumulator_value, new_accumulators)


    # Main loop logic
    def cache_update(self,
                          name: str,
                          item: PyTree
                          )->'ACT_Controller':
        """
        Cache a new group of accumulator pytrees onto a configured
        entry. Caching will save the tree for the later weight-and-add process.

        :param name: The name of the accumulator to cache
        :param item: What to actually cache
        :except: ValueError, if accumulator referenced does not exist
        :except: RuntimeError, if accumulator was already set.
        :return: New ACT_Controller with updated state
        """
        self._check_if_locked()
        state = self.state
        if name not in state.updates:
            raise ValueError(f"Accumulator with name {name} was never setup")
        if state.updates[name] is not None:
            raise RuntimeError(f"Accumulator with name {name} was already set this iteration!")
        update = state.updates.copy()
        update[name] = item
        new_state = state.replace(updates=update)
        return ACT_Controller(new_state)
    def iterate_act(self,
                    halting_probabilities: jnp.ndarray
                    )->'ACT_Controller':
        """
        Iterates the act process. Returns the result

        :param halting_probabilities:
        :return: The act controller with the iteration performed.
        """
        if halting_probabilities.shape != self.probabilities.shape:
            raise ValueError("provided and original shapes of halting probabilities do not match")

        self._check_if_locked()
        # Compute and manage probability quantities.
        #
        # This includes clamping the halting probabilities, and
        # making the new probabilities and residuals.

        halting_probabilities, probabilities, residuals = self._process_probabilities(halting_probabilities)

        # Perform updates on each accumulator. Fetch the
        # update, run the update method, store it.

        accumulators = self.accumulators.copy()
        updates = self.state.updates.copy()
        for name in accumulators.keys():
            try:
                accumulator_tree = accumulators[name]
                update_tree = updates[name]

                # To update the accumulators, which may be pytrees, we
                # create an update function to apply update logic on
                # accumulator, update leaf pairs. Then we use
                # merge pytrees to zip the trees together.

                update_function = lambda accumulator_leaf, update_leaf : self._update_accumulator(
                                                                            accumulator_leaf,
                                                                            update_leaf,
                                                                            halting_probabilities
                                                                            )
                new_accumulator = self._merge_pytrees(update_function,
                                                      accumulator_tree,
                                                      update_tree
                                                    )

                accumulators[name] = new_accumulator
                updates[name] = None # Resets update slot, so we do not think an update was already done.
            except Exception as err:
                msg = f"An error occurred regarding accumulator {name}: \n\n"
                raise RuntimeError(msg) from err

        # We update the iteration count only where we have
        # not reached the halting state

        new_iterations = self.iterations + 1
        iterations = jnp.where(self.halted_batches, self.iterations, new_iterations)

        # Create the updated state. Return.

        state = self.state.replace(
            iterations = iterations,
            residuals = residuals,
            probabilities=probabilities,
            accumulators=accumulators,
            updates = updates,
        )
        return ACT_Controller(state)

    ## Display functions
    #
    # These are used primarily to make it easy to look at
    # and use your gathered data and statistics.
    def mask_data(self,
                  mask: jnp.ndarray,
                  lock: bool = True,
                  )->'ACT_Controller':
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


        iterations = self.iterations*mask
        residuals = self.residuals*mask
        probabilities = self.probabilities*mask

        #TODO: Fix

        update_func = lambda tree : self._apply_mask(mask, tree)
        accumulators = {name : jax.tree_util.tree_map(update_func, value)
                        for name, value
                        in self.accumulators.items()}

        state = self.state.replace(is_locked=lock,
                                   iterations=iterations,
                                   residuals = residuals,
                                   probabilities = probabilities,
                                   accumulators= accumulators
                                   )

        return ACT_Controller(state)
    def mask_unhalted_data(self)->'ACT_Controller':
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

    def reset_batches(self)->'ACT_Controller':
        """
        Resets the batches that are currently halted. Accumulators
        are set to their default values, and probabilities are set
        to zero.

        :return: An ACT_Controller instance with halted batches
                 reset to default
        """
        self._check_if_locked()

        unhalted_batch_selector = ~self.halted_batches

        # Multiplying by zero resets the probabilities and iteration statistics.

        iterators = self.iterations*unhalted_batch_selector
        probabilities = self.probabilities*unhalted_batch_selector
        residuals = self.residuals*unhalted_batch_selector

        accumulators = {}
        for name in self.accumulators.keys():
            accumulator = self.accumulators[name]
            default = self.state.defaults[name]

            # To merge the pytrees, we use a helper function that applies
            # a update function to pairs of like leaves.
            update_func = lambda accumulator, default : jnp.where(unhalted_batch_selector,
                                                                  accumulator,
                                                                  default
                                                                  )
            new_accumulator = self._merge_pytrees(update_func,
                                                  accumulator,
                                                  default)
            accumulators[name] = new_accumulator

        # Create the new state, return the new
        # controller

        state = self.state.replace(
            residuals=residuals,
            probabilities=probabilities,
            iterations=iterators,
            accumulators=accumulators
        )

        return ACT_Controller(state)
    # Saving, loading, and pytrees
    def save(self)->ACTStates:
        return self.state
    @classmethod
    def load(cls, state: ACTStates)->'ACT_Controller':
        return cls(state)
    def __init__(self, state: ACTStates):
        super().__init__()
        self.state = state
        self.make_immutable()

def flatten_controller(controller: ACT_Controller)->Tuple[ACTStates, Any]:
    state = controller.save()
    return state, None

def unflatten_controller(aux: Any, state: ACTStates)->ACT_Controller:
    return ACT_Controller.load(state)

jax.tree_util.register_pytree_node(ACT_Controller, flatten_controller, unflatten_controller)