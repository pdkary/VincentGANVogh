from dataclasses import dataclass, field
from typing import List, Tuple
from config.CallableConfig import ActivationConfig, NoneCallable, RegularizationConfig
from config.GanConfig import ConvLayerConfig
from inputs.GanInput import GanInput

std_dim_factory = lambda : [1,2,3]

@dataclass
class GANBase():
    gan_input:GanInput
    conv_layers: List[ConvLayerConfig]
    dense_layers: List[int]
    conv_input_shape: Tuple[int,int,int]
    view_layers: bool = False
    std_dims: List[int] = field(default_factory=std_dim_factory)
    minibatch_size: int = 0
    dropout_rate: float = 0.0
    dense_activation: ActivationConfig = field(default=NoneCallable)
    final_activation: ActivationConfig = field(default=NoneCallable)
    view_activation: ActivationConfig = field(default=NoneCallable)
    kernel_regularizer: RegularizationConfig = field(default=NoneCallable)
    kernel_initializer: str = field(default="glorot_uniform")
    tracked_layers = {}
    viewing_layers = []

    @property
    def input(self):
        return self.gan_input.input_layer

    def get_training_batch(self,batch_size):
        b = [self.gan_input.get_training_batch(batch_size)]
        return b

    def get_validation_batch(self,batch_size):
        b = [self.gan_input.get_validation_batch(batch_size)]
        return b
