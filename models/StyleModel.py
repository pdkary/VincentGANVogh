from typing import Tuple
from inputs.GanInput import GanInput

from layers.CallableConfig import ActivationConfig
from models.DenseModel import DenseModel

class LatentStyleModel(DenseModel):
    def __init__(self,
                 style_input: GanInput,
                 style_layers: int,
                 style_layer_size: int,
                 activation: ActivationConfig):
        super().__init__(style_input, [style_layer_size for i in range(style_layers)], activation)
