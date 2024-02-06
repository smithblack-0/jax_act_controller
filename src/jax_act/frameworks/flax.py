import textwrap

from flax import linen as nn
from src.jax_act.layers.act import AbstractACTTemplate
from src.jax_act import ACT_Controller, PyTree

class FlaxACTLayer(AbstractACTTemplate, nn.Module):
    """


    """
    #TODO: Make docstring for class

    ## Fufill portions of the contract
    #
    # We fufill the setup_state promise, and
    # also instrument setup with some error messages
    #
    # We do NOT configure run_iteration or
    # make_controller. Those are the responsibilities
    # of subclasses.
    def setup(self):
        msg = """
        It seems like method "setup" was not implemented in your 
        act layer.  Inline definitions, using @nn.compact, 
        do not currently work. They break flax in some horrible
        way. 
        
        Instead, you must define your layers in setup then use them 
        in your call. However, you may feel free to use modules which
        contain inline definitions. See for more details:
        
        https://flax.readthedocs.io/en/latest/guides/flax_fundamentals/setup_or_nncompact.html
        
        If someone smarter than me wants to fix that, they are free to.
        Until then, use setup to define your layers rather than
        inline creation.
        """
        msg = textwrap.dedent(msg)
        raise NotImplementedError(msg)

    def setup_lazy_parameters(self, controller: ACT_Controller, state: PyTree):
        # We run a single iteration if the controller's
        # parameters are not yet immutable.
        if self.is_mutable_collection("params"):
            self.run_iteration(controller, state)
