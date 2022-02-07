from dataclasses import dataclass, field
from typing import List, Tuple

from config.CallableConfig import (ActivationConfig, NoneCallable,
                                   RegularizationConfig)
from config.DNA import GanDNA
from config.GanConfig import (ConvLayerConfig, DenseLayerConfig,
                              SimpleActivations)
from helpers import SearchableEnum
from inputs.GanInput import GanInput


@dataclass
class GANBase():
    gan_input:GanInput
    conv_layers: List[ConvLayerConfig]
    dense_layers: List[DenseLayerConfig]
    conv_input_shape: Tuple[int,int,int]
    view_layers: bool = False
    std_dims: List[int] = field(default_factory = lambda : [1,2,3])
    view_activation: ActivationConfig = field(default=NoneCallable)
    kernel_regularizer: RegularizationConfig = field(default=NoneCallable)
    kernel_initializer: str = field(default="glorot_uniform")
    tracked_layers = {}
    viewing_layers = []

    @property
    def input(self):
        return self.gan_input.input_layer

    @property
    def has_tracked_layers(self):
        return any([x.track_id != "" for x in self.conv_layers])
    
    @property
    def num_tracked_layers(self):
        return sum([x.track_id != "" for x in self.conv_layers])

    def get_training_batch(self,batch_size):
        b = [self.gan_input.get_training_batch(batch_size)]
        return b

    def get_validation_batch(self,batch_size):
        b = [self.gan_input.get_validation_batch(batch_size)]
        return b
    
    def to_DNA(self,activation_set: SearchableEnum = SimpleActivations):
        d_acts = [d.activation for d in self.dense_layers]
        d_acts_indices = [activation_set.get_index_by_name(d.name) for d in d_acts]

        c_acts = [c.activation for c in self.conv_layers]
        c_acts_indices = [activation_set.get_index_by_name(d.name) for d in c_acts]
        return GanDNA([d.size for d in self.dense_layers],
                      d_acts_indices,
                      [d.minibatch_size for d in self.dense_layers], 
                      [d.minibatch_dim for d in self.dense_layers], 
                      [d.dropout_rate for d in self.dense_layers], 
                      self.conv_input_shape,
                      [x.filters for x in self.conv_layers],
                      [x.convolutions for x in self.conv_layers],
                      [x.kernel_size for x in self.conv_layers],
                      [x.upsampling for x in self.conv_layers],
                      [x.downsampling for x in self.conv_layers],
                      c_acts_indices,
                      [x.noise for x in self.conv_layers],
                      activation_set,
                      self.kernel_regularizer,
                      self.kernel_initializer)
    
    @staticmethod
    def from_DNA(gan_input: GanInput,dna: GanDNA):
        conv_activations = [dna.activation_set.get_value_by_index(i) for i in dna.conv_activations]
        conv_data = zip(dna.conv_filters,
                        dna.conv_convolutions,
                        dna.conv_kernel_sizes,
                        dna.conv_upsampling,
                        dna.conv_downsampling,
                        conv_activations,
                        dna.conv_noises)
        conv_layers = [ConvLayerConfig(f,c,k,a,upsampling=u,downsampling=d,noise=n) for f,c,k,u,d,a,n in conv_data]

        dense_activations = [dna.activation_set.get_value_by_index(i) for i in dna.dense_activations]
        dense_data = zip(dna.dense_layers,
                         dense_activations,
                         dna.dense_dropout,
                         dna.dense_minibatch_size,
                         dna.dense_minibatch_dim)
        dense_layers = [DenseLayerConfig(s,a,dr,ms,md) for s,a,dr,ms,md in dense_data]
        
        return GANBase(gan_input,
                       conv_layers,
                       dense_layers,
                       conv_input_shape=dna.conv_input_shape,
                       kernel_regularizer=dna.kernel_regularizer,
                       kernel_initializer=dna.kernel_initializer)
