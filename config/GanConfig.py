from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Union

from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.regularizers import L2
from third_party_layers import InstanceNormalization

from config.CallableConfig import (ActivationConfig, NoneCallable,
                                   NormalizationConfig, RegularizationConfig)


class SimpleActivations(Enum):
    ##acts
    sigmoid = ActivationConfig(Activation,dict(activation="sigmoid"))
    softmax = ActivationConfig(Activation,dict(activation="softmax"))
    tanh = ActivationConfig(Activation,dict(activation="tanh"))
    relu = ActivationConfig(Activation,dict(activation="relu"))
    linear = ActivationConfig(Activation,dict(activation="linear"))
    ##norms
    instance_norm = NormalizationConfig(InstanceNormalization)
    batch_norm = NormalizationConfig(BatchNormalization,dict(momentum=0.8))
    ##regularizer
    l2 = RegularizationConfig(L2)

@dataclass
class DiscDenseLayerConfig():
    size: int
    activation: ActivationConfig
    dropout_rate: int

@dataclass
class ConvLayerConfig:
    filters: int
    convolutions: int
    kernel_size: int
    activation: ActivationConfig
    strides: Tuple[int,int] = (1,1)
    transpose: bool = False
    upsampling: Union[bool,str] = False
    downsampling: Union[bool,str] = False
    style: bool = False
    noise: float = 0.0
    dropout_rate: float = 0.0
    normalization: NormalizationConfig = NoneCallable
    regularizer: RegularizationConfig = NoneCallable
    kernel_initializer: str = "glorot_uniform"
    track_id: str = ""

    def flip(self):
        u = self.upsampling
        d = self.downsampling
        self.upsampling = d
        self.downsampling = u
        return self

@dataclass
class DiscConvLayerConfig(ConvLayerConfig):
    downsampling: Union[bool,str] = True

@dataclass
class GenLayerConfig(ConvLayerConfig):
    pass
