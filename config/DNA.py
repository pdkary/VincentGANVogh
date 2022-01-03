from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from config.CallableConfig import ActivationConfig, NoneCallable
from config.GanConfig import SimpleActivations


@dataclass
class GanDNA():
    dense_layers: List[int]
    conv_input_shape: Tuple[int, int, int]
    conv_filters: List[int]
    conv_layers: List[int]
    conv_kernel_sizes: List[int]
    conv_upsampling: List[bool] = field(default_factory=list)
    dense_activation: ActivationConfig = field(default=NoneCallable)
    conv_activation: ActivationConfig = field(default=NoneCallable)
    final_activation: ActivationConfig = field(default=SimpleActivations.sigmoid.value)
    kernel_regularizer: ActivationConfig = field(default=NoneCallable)
    kernel_initializer: str = field(default="glorot_uniform")
    minibatch_size: int = 0
    dropout_rate: float = 0.0

    @property
    def conv_matrix(self):
        return np.stack((self.conv_filters, self.conv_layers, self.conv_kernel_sizes, self.conv_upsampling))

    @property
    def size_factor(self):
        return 2**np.sum(self.conv_upsampling)


@dataclass
class DiscriminatorDNA(GanDNA):
    @property
    def output_shape(self):
        return self.dense_layers[-1]

    @property
    def dense_input_shape(self):
        df = self.size_factor
        return (self.conv_input_shape[0]//df, self.conv_input_shape[1]//df, self.conv_filters[-1])

    def summary(self):
        print(self.conv_input_shape)
        print(self.conv_matrix)
        print(self.dense_input_shape)
        print(self.dense_layers)
        print(self.output_shape)

    def to_gen_DNA(self):
        conv_filters = list(reversed(deepcopy(self.conv_filters)))
        conv_layers = list(reversed(deepcopy(self.conv_layers)))
        conv_kernel = list(reversed(deepcopy(self.conv_kernel_sizes)))
        conv_up = list(reversed(deepcopy(self.conv_upsampling)))
        dense_layers = list(reversed(deepcopy(self.dense_layers)))

        return GeneratorDNA(dense_layers, self.dense_input_shape, conv_filters,
                            conv_layers, conv_kernel, conv_up,
                            self.dense_activation, self.conv_activation,
                            self.final_activation,
                            self.kernel_regularizer, self.kernel_initializer)


@dataclass
class GeneratorDNA(GanDNA):
    @property
    def output_shape(self):
        uf = self.size_factor
        return (uf*self.conv_input_shape[0], uf*self.conv_input_shape[1], self.conv_filters[-1])

    def summary(self):
        print(self.dense_layers)
        print(self.conv_input_shape)
        print(self.conv_matrix)
        print(self.output_shape)

    def validate(self, output_shape):
        # assert matching layer lengths
        assert len(self.conv_filters) == len(self.conv_layers)
        assert len(self.conv_layers) == len(self.conv_kernel_sizes)

        if self.conv_upsampling == []:
            self.conv_upsampling = [False]*len(self.conv_layers)
        else:
            assert len(self.conv_kernel_sizes) == len(self.conv_upsampling)

        # assert output shapes
        assert self.output_shape == output_shape

    def to_disc_DNA(self):
        conv_filters = list(reversed(deepcopy(self.conv_filters)))
        conv_layers = list(reversed(deepcopy(self.conv_layers)))
        conv_kernel = list(reversed(deepcopy(self.conv_kernel_sizes)))
        conv_up = list(reversed(deepcopy(self.conv_upsampling)))
        dense_layers = list(reversed(deepcopy(self.dense_layers)))
        return DiscriminatorDNA(dense_layers, self.output_shape, conv_filters,
                                conv_layers, conv_kernel, conv_up,
                                self.dense_activation, self.conv_activation,
                                self.final_activation,
                                self.kernel_regularizer,
                                self.kernel_initializer)
