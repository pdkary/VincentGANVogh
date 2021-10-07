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
                 disc_conv_layers: List[DiscConvLayerConfig],
                 disc_dense_layers: List[int],
                 minibatch_size: int = 0,
                 dropout_rate: float = 0.0,
                 activation: ActivationConfig = NoneCallable,
                 kernel_regularizer: RegularizationConfig = NoneCallable,
                 kernel_initializer: str = "glorot_uniform"):
        self.gan_input = real_image_input
        
        self.CM = ConvolutionalModel(gan_input=real_image_input,
                                             conv_layers=disc_conv_layers,
                                             kernel_regularizer=kernel_regularizer,
                                             kernel_initializer=kernel_initializer)

        self.conv_out = self.CM.build(flatten=True)
        # self.internal_input = Input(shape=self.conv_out.shape[-1],dtype=tf.float32)
        # print(self.internal_input)
        
        self.DM = DenseModel(input=self.conv_out,
                                      dense_layers=disc_dense_layers,
                                      activation=activation,
                                      minibatch_size=minibatch_size,
                                      dropout_rate=dropout_rate)

        self.functional_model = self.DM.build()
        self.tracked_layers = self.CM.tracked_layers

