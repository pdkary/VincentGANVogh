from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from config.CallableConfig import ActivationConfig, NoneCallable
from config.GanConfig import SimpleActivations
from helpers.SearchableEnum import SearchableEnum

@dataclass
class GanDNA():
    #dense layers
    dense_layers: List[int]
    dense_activations: List[int]
    dense_minibatch_size: List[int]
    dense_minibatch_dim: List[int]
    dense_dropout: List[float]
    conv_input_shape: Tuple[int, int, int]
    ##conv matrix
    conv_filters: List[int]
    conv_convolutions: List[int]
    conv_kernel_sizes: List[int]
    conv_upsampling: List[bool]
    conv_downsampling: List[bool]
    conv_activations: List[int]
    conv_noises: List[int]
    ## map for activations
    activation_set: SearchableEnum = field(default=SimpleActivations)
    ##additional data
    kernel_regularizer: ActivationConfig = field(default=NoneCallable)
    kernel_initializer: str = field(default="glorot_uniform")

    @property
    def dense_matrix(self):
        return np.stack((self.dense_layers,
                         self.dense_activations,
                         self.dense_minibatch_size,
                         self.dense_minibatch_dim,
                         self.dense_dropout))

    @property
    def conv_matrix(self):
        return np.stack((self.conv_filters, 
                         self.conv_convolutions, 
                         self.conv_kernel_sizes, 
                         self.conv_upsampling,
                         self.conv_downsampling,
                         self.conv_activations,
                         self.conv_noises))

    @property
    def up_factor(self):
        return 2**np.sum(self.conv_upsampling)
    
    @property
    def down_factor(self):
        return 2**np.sum(self.conv_downsampling)

    @property
    def conv_output_shape(self):
        sf = self.up_factor / self.down_factor
        return (int(sf*self.conv_input_shape[0]), int(sf*self.conv_input_shape[1]), self.conv_filters[-1])


@dataclass
class DiscriminatorDNA(GanDNA):
    def summary(self):
        print("Input Shape: \n",self.conv_input_shape)
        print("Conv Matrix: \n",self.conv_matrix)
        print("Conv Output Shape: \n",self.conv_output_shape)
        print("Dense Matrix: \n",self.dense_matrix)
        print("Output Shape: \n",self.dense_layers[-1])

    def to_gen_DNA(self):
        conv_filters = list(reversed(deepcopy(self.conv_filters)))
        conv_layers = list(reversed(deepcopy(self.conv_convolutions)))
        conv_kernel = list(reversed(deepcopy(self.conv_kernel_sizes)))
        conv_up = list(reversed(deepcopy(self.conv_downsampling)))
        conv_down = list(reversed(deepcopy(self.conv_upsampling)))
        conv_acts = list(reversed(deepcopy(self.conv_activations)))
        conv_noise = list(reversed(deepcopy(self.conv_noises)))

        dense_layers = list(reversed(deepcopy(self.dense_layers)))
        dense_acts = list(reversed(deepcopy(self.dense_activations)))
        dense_mini_s = list(reversed(deepcopy(self.dense_minibatch_size)))
        dense_mini_d = list(reversed(deepcopy(self.dense_minibatch_dim)))
        dense_dr = list(reversed(deepcopy(self.dense_dropout)))
        return GeneratorDNA(dense_layers, dense_acts, dense_mini_s,dense_mini_d,dense_dr,
                            self.conv_output_shape, 
                            conv_filters, conv_layers, conv_kernel, conv_up, conv_down, conv_acts, conv_noise,
                            self.activation_set,
                            self.kernel_regularizer, 
                            self.kernel_initializer)


@dataclass
class GeneratorDNA(GanDNA):
    def summary(self):
        print("Dense Matrix: \n",self.dense_matrix)
        print("Conv input shape: \n",self.conv_input_shape)
        print("Conv Matrix: \n",self.conv_matrix)
        print("Conv Output Shape: \n",self.conv_output_shape)

    def validate(self, output_shape):
        # assert matching layer lengths
        assert len(self.conv_filters) == len(self.conv_convolutions)
        assert len(self.conv_convolutions) == len(self.conv_kernel_sizes)

        if self.conv_upsampling == []:
            self.conv_upsampling = [False]*len(self.conv_convolutions)
        else:
            assert len(self.conv_kernel_sizes) == len(self.conv_upsampling)

        # assert output shapes
        assert self.conv_output_shape == output_shape

    def to_disc_DNA(self):
        conv_filters = list(reversed(deepcopy(self.conv_filters)))
        conv_layers = list(reversed(deepcopy(self.conv_convolutions)))
        conv_kernel = list(reversed(deepcopy(self.conv_kernel_sizes)))
        conv_up = list(reversed(deepcopy(self.conv_downsampling)))
        conv_down = list(reversed(deepcopy(self.conv_upsampling)))
        conv_act = list(reversed(deepcopy(self.conv_activations)))
        conv_noise = list(reversed(deepcopy(self.conv_noises)))

        dense_layers = list(reversed(deepcopy(self.dense_layers)))
        dense_acts = list(reversed(deepcopy(self.dense_activations)))
        dense_mini_s = list(reversed(deepcopy(self.dense_minibatch_size)))
        dense_mini_d = list(reversed(deepcopy(self.dense_minibatch_dim)))
        dense_dr = list(reversed(deepcopy(self.dense_dropout)))

        return DiscriminatorDNA(dense_layers, dense_acts, dense_mini_s, dense_mini_d,dense_dr,
                                self.conv_output_shape, 
                                conv_filters,conv_layers,conv_kernel,conv_up,conv_down,conv_act,conv_noise, 
                                self.activation_set,
                                self.kernel_regularizer,
                                self.kernel_initializer)
