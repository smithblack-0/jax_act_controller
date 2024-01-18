from src.immutable import Immutable
from src.states import ACTStates, ViewerConfig


class ACT_Viewer(Immutable):

    def _apply_mask(self, ):


    @properties
    def probabilities(self):


    def __init__(self,
                 state: ACTStates,
                 configuration: ViewerConfig):
        super().__init__()
        self.states = state
        self.make_immutable()



