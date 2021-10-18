from typing import List

import tensorflow as tf
from config.GanConfig import DiscConvLayerConfig
from inputs.GanInput import GanInput
from layers.CallableConfig import (ActivationConfig, NoneCallable,
                                   RegularizationConfig)
from tensorflow.keras.layers import Input

from models.ConvolutionalModel import ConvolutionalModel
from models.DenseModel import DenseModel


class Discriminator():
    def __init__(self,
                 real_image_input: GanInput,
                 conv_layers: List[DiscConvLayerConfig],
                 dense_layers: List[int],
                 minibatch_size: int = 0,
                 dropout_rate: float = 0.0,
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
        self.dense_activation = dense_activation
        self.final_activation = final_activation
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
    
    def build(self):
        CM = ConvolutionalModel(self.input,
                                self.conv_layers,
                                None,
                                self.kernel_regularizer,
                                self.kernel_initializer)

        self.conv_out = CM.build(flatten=True)
        
        DM = DenseModel(self.conv_out,self.dense_layers,
                        self.dense_activation,self.minibatch_size,
                        self.dropout_rate)
        DM_out = DM.build()
        self.dense_out = self.final_activation.get()(DM_out)
        self.tracked_layers = CM.tracked_layers
        return self.dense_out

    def get_training_batch(self,batch_size):
        b = [self.gan_input.get_training_batch(batch_size)]
        return b

    def get_validation_batch(self,batch_size):
        b = [self.gan_input.get_validation_batch(batch_size)]
        return b
