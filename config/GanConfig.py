from dataclasses import dataclass
from typing import Iterable, List, Tuple, Union

from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import L2
from helpers.SearchableEnum import SearchableEnum
from layers.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization
from third_party_layers.InstanceNormalization import InstanceNormalization

from config.CallableConfig import (ActivationConfig, NoneCallable,
                                   NormalizationConfig, RegularizationConfig)


class SimpleActivations(SearchableEnum):
    sigmoid       = ActivationConfig(Activation, [], dict(activation="sigmoid"))
    softmax       = ActivationConfig(Activation, [], dict(activation="softmax"))
    tanh          = ActivationConfig(Activation, [], dict(activation="tanh"))
    relu          = ActivationConfig(Activation, [], dict(activation="relu"))
    linear        = ActivationConfig(Activation, [], dict(activation="linear"))
    leakyRelu_p08 = ActivationConfig(LeakyReLU,  [], dict(alpha=0.08))
    leakyRelu_p1  = ActivationConfig(LeakyReLU,  [], dict(alpha=0.1))

class SimpleNormalizations(SearchableEnum):
    instance_norm = NormalizationConfig(InstanceNormalization,[])
    batch_norm    = NormalizationConfig(BatchNormalization,[],dict(momentum=0.8))
    batch_norm_factory = lambda m: NormalizationConfig(BatchNormalization,[],dict(momentum=m))
    adain_factory = lambda s=None: NormalizationConfig(AdaptiveInstanceNormalization,[])

class SimpleRegularizers(SearchableEnum):
    l2 = RegularizationConfig(L2)

@dataclass
class DenseLayerConfig():
    size: int
    activation: ActivationConfig
    dropout_rate: float = 0
    minibatch_size: int = 0
    minibatch_dim: int = 0

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
    concat_with: str = ""

    def flip(self):
        u = self.upsampling
        d = self.downsampling
        self.upsampling = d
        self.downsampling = u
        return self

@dataclass
class DiscConvLayerConfig(ConvLayerConfig):
    pass

@dataclass
class GenLayerConfig(ConvLayerConfig):
    pass

@dataclass
class TrackedLayerConfig:
    std: List
    mean: List
    view = None