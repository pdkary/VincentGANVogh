from typing import Tuple
from layers.CallableConfig import ActivationConfig, NoneCallable, NormalizationConfig

class DiscConvLayerConfig():
    def __init__(self,
                 filters: int,
                 convolutions: int,
                 kernel_size: int,
                 dropout_rate: float = 0.5,
                 downsampling: bool = True,
                 normalization: NormalizationConfig = NoneCallable):
        self.filters = filters
        self.convolutions = convolutions
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.downsampling = downsampling
        self.normalization = normalization

class DiscDenseLayerConfig():
    def __init__(self,
                 size: int,
                 activation: ActivationConfig,
                 dropout_rate: int):
        self.size = size
        self.activation = activation
        self.dropout_rate = dropout_rate

class GenLayerConfig():
    def __init__(self,
                 filters: int,
                 convolutions: int,
                 kernel_size: int,
                 strides: Tuple[int, int] = (1, 1),
                 transpose: bool = False,
                 upsampling: bool = False,
                 style: bool = False,
                 noise: bool = False):
        self.filters = filters
        self.convolutions = convolutions
        self.kernel_size = kernel_size
        self.strides = strides
        self.transpose = transpose
        self.upsampling = upsampling
        self.style = style
        self.noise = noise