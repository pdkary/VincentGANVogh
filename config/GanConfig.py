from typing import Tuple, Union, List
from layers.CallableConfig import ActivationConfig, NoneCallable, NormalizationConfig, RegularizationConfig
import numpy as np
from dataclasses import dataclass, field

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

def breed_arrays(arr1,arr2,prob1=0.5):
    if len(arr1) > len(arr2):
        arr2 = arr2 + [0]*(len(arr1)-len(arr2))
    elif len(arr1) < len(arr2):
        arr1 = arr2 + [0]*(len(arr2)-len(arr1))
    
    arr_out = []
    for x1,x2 in zip(arr1,arr2):
        x = x1 if np.random.rand() <= prob1 else x2
    
    return [x for x in arr_out if x >= 1]         

@dataclass
class GeneratorDNA():
    dense_layers: List[int]
    conv_input_shape: Tuple[int,int,int]
    conv_sizes: List[int]
    conv_layers: List[int]
    conv_kernel_sizes: List[int]
    conv_upsampling: List[bool] = field(default_factory=list)

    @property
    def output_shape(self):
        up_factor = 2**np.sum(self.conv_upsampling)
        return (up_factor*self.conv_input_shape[0],up_factor*self.conv_input_shape[1],self.conv_sizes[-1])

    def breed(self, genDNA):
        new_dense = breed_arrays(self.dense_layers,genDNA.dense_layers)
        new_sizes = breed_arrays(self.conv_sizes,genDNA.conv_sizes)
        new_layers = breed_arrays(self.conv_layers,genDNA.conv_layers)
        new_kernels = breed_arrays(self.conv_kernel_sizes,genDNA.conv_kernel_sizes)
        new_ups = breed_arrays(self.conv_upsampling,genDNA.conv_upsampling)

        return GeneratorDNA(new_dense,self.conv_input_shape,
                            new_sizes,new_layers,
                            new_kernels,new_ups)    

    def validate(self,output_shape):
        ##assert matching layer lengths
        assert len(self.conv_sizes) == len(self.conv_layers)
        assert len(self.conv_layers) == len(self.conv_kernel_sizes)

        if self.conv_upsampling == []:
            self.conv_upsampling = [False]*len(self.conv_layers)
        else:
            assert len(self.conv_kernel_sizes) == len(self.conv_upsampling)

        ##assert output shapes
        assert self.output_shape == output_shape