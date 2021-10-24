from typing import List
from copy import deepcopy
import tensorflow as tf
from config.GanConfig import DiscConvLayerConfig
from inputs.GanInput import GanInput
from layers.CallableConfig import (ActivationConfig, NoneCallable,
                                   RegularizationConfig)
from tensorflow.keras.layers import Input

from models.ConvolutionalModel import ConvolutionalModel
from models.DenseModel import DenseModel
from models.Generator import Generator


class Discriminator():
    @staticmethod
    def from_generator(generator:Generator,
                       real_image_input: GanInput,
                       final_activation: ActivationConfig,
                       output_dim: int = None,
                       minibatch_size: int = 0,
                       dropout_rate: float = 0.0,
                       viewable: bool = False):
        conv_layers = [x.flip() for x in reversed(deepcopy(generator.conv_layers))]
        dense_layers = list(reversed(deepcopy(generator.dense_layers)))
        dense_layers.append(generator.gan_input.input_shape[-1])
        dense_layers.append(output_dim)
        dense_activation = generator.dense_activation
        kr = generator.kernel_regularizer
        ki = generator.kernel_initializer
        return Discriminator(real_image_input,conv_layers,dense_layers,
                             minibatch_size=minibatch_size,
                             dropout_rate=dropout_rate,
                             view_layers=viewable,
                             dense_activation=dense_activation,
                             final_activation=final_activation,
                             kernel_regularizer=kr,
                             kernel_initializer=ki)

    def __init__(self,
                 real_image_input: GanInput,
                 conv_layers: List[DiscConvLayerConfig],
                 dense_layers: List[int],
                 minibatch_size: int = 0,
                 dropout_rate: float = 0.0,
                 view_layers: bool = False,
                 dense_activation: ActivationConfig = NoneCallable,
                 final_activation: ActivationConfig = NoneCallable,
                 kernel_regularizer: RegularizationConfig = NoneCallable,
                 kernel_initializer: str = "glorot_uniform"):
        self.gan_input: GanInput = real_image_input
        self.input = real_image_input.input_layer
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.minibatch_size = minibatch_size
        self.dropout_rate = dropout_rate
        self.view_layers = view_layers
        self.dense_activation = dense_activation
        self.final_activation = final_activation
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.viewing_layers = []
    
    def build(self):
        print("BUILDING DISCRIMINATOR")
        print("BUILDING DISCRIMINATOR CONV MODEL")
        channels = self.gan_input.input_shape[-1] if self.view_layers else None
        CM = ConvolutionalModel(self.input,
                                self.conv_layers,
                                None,
                                channels,
                                self.kernel_regularizer,
                                self.kernel_initializer)

        self.conv_out = CM.build(flatten=True)
        
        print("BUILDING DISCRIMINATOR DENSE MODEL")
        DM = DenseModel(self.conv_out,self.dense_layers,
                        self.dense_activation,self.minibatch_size,
                        self.dropout_rate)
        DM_out = DM.build()
        self.dense_out = self.final_activation.get()(DM_out)
        self.tracked_layers = CM.tracked_layers
        self.viewing_layers = CM.viewing_layers
        return self.dense_out

    def get_training_batch(self,batch_size):
        b = [self.gan_input.get_training_batch(batch_size)]
        return b

    def get_validation_batch(self,batch_size):
        b = [self.gan_input.get_validation_batch(batch_size)]
        return b
