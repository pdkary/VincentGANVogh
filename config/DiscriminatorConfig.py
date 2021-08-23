from config.CallableConfig import ActivationConfig, NormalizationConfig
from typing import Tuple, List
from keras.optimizers import Optimizer
    
class DiscConvLayerConfig():
  def __init__(self,
               filters: int,
               convolutions: int,
               kernel_size: int,
               activation_config: ActivationConfig,
               normalization: NormalizationConfig):
    self.filters = filters
    self.convolutions = convolutions
    self.kernel_size = kernel_size
    self.activation_config = activation_config
    self.normalization = normalization

class DiscDenseLayerConfig():
  def __init__(self,
               size: int,
               activation_config: ActivationConfig,
               dropout_rate: int):
    self.size = size
    self.activation_config = activation_config
    self.dropout_rate = dropout_rate

class DiscriminatorModelConfig():
  def __init__(self,
               img_shape: Tuple[int,int,int],
               disc_conv_layers: List[DiscConvLayerConfig],
               disc_dense_layers: List[DiscDenseLayerConfig],
               minibatch: int,
               minibatch_size: int,
               disc_optimizer: Optimizer):
    self.img_shape = img_shape
    self.disc_conv_layers = disc_conv_layers
    self.disc_dense_layers = disc_dense_layers
    self.minibatch = minibatch
    self.minibatch_size = minibatch_size
    self.disc_optimizer = disc_optimizer
