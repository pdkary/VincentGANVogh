from typing import Tuple
from layers.CallableConfig import ActivationConfig, NoneCallable, NormalizationConfig, RegularizationConfig

class DiscDenseLayerConfig():
    def __init__(self,
                 size: int,
                 activation: ActivationConfig,
                 dropout_rate: int):
        self.size = size
        self.activation = activation
        self.dropout_rate = dropout_rate

class ConvLayerConfig():
    def __init__(self,
                 filters: int,
                 convolutions: int,
                 kernel_size: int,
                 activation: ActivationConfig,
                 strides: Tuple[int,int] = (1,1),
                 transpose: bool = False,
                 upsampling: bool = False,
                 downsampling: bool = False,
                 normalization: NormalizationConfig = NoneCallable,
                 regularizer: RegularizationConfig = NoneCallable,
                 kernel_initializer: str = "glorot_uniform",
                 track_id: str = None):
        self.filters = filters 
        self.convolutions = convolutions
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        self.transpose = transpose
        self.upsampling = upsampling
        self.downsampling = downsampling
        self.normalization = normalization 
        self.regularizer = regularizer
        self.kernel_initializer = kernel_initializer
        self.track_id = track_id
        
class DiscConvLayerConfig(ConvLayerConfig):
    def __init__(self,
                 filters: int,
                 convolutions: int,
                 kernel_size: int,
                 activation: ActivationConfig,
                 dropout_rate: float = 0.5,
                 downsampling: bool = True,
                 normalization: NormalizationConfig = NoneCallable,
                 regularizer: RegularizationConfig = NoneCallable,
                 track_id: str = None):
        super().__init__(filters,convolutions,kernel_size,activation=activation,downsampling=downsampling,
                         normalization=normalization,regularizer=regularizer, track_id=track_id)
        self.dropout_rate = dropout_rate
        
class GenLayerConfig(ConvLayerConfig):
    def __init__(self,
                 filters: int,
                 convolutions: int,
                 kernel_size: int,
                 activation: ActivationConfig,
                 strides: Tuple[int, int] = (1, 1),
                 transpose: bool = False,
                 upsampling: bool = False,
                 style: bool = False,
                 noise: bool = False,
                 track_id: str = None):
        super().__init__(filters,convolutions,kernel_size,activation=activation,
                         strides=strides,transpose=transpose,upsampling=upsampling,
                         track_id=track_id)
        self.style = style
        self.noise = noise