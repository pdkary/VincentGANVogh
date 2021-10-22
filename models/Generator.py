from typing import List, Tuple

from tensorflow.keras.layers import Dense, Reshape
from config.GanConfig import GenLayerConfig
from inputs.GanInput import GanInput
from layers.CallableConfig import ActivationConfig, NoneCallable, RegularizationConfig

from models.ConvolutionalModel import ConvolutionalModel
from models.DenseModel import DenseModel
import numpy as np

class Generator():
    def __init__(self,
                 gan_input: GanInput,
                 dense_layers: List[int],
                 conv_input_shape: Tuple[int],
                 conv_layers: List[GenLayerConfig],
                 style_input: GanInput = None,
                 style_layers: List[int] = [],
                 view_layers: bool = False,
                 dense_activation: ActivationConfig = NoneCallable,
                 kernel_regularizer:RegularizationConfig = NoneCallable,
                 kernel_initializer:str = "glorot_uniform"):
        self.gan_input = gan_input
        self.dense_layers = dense_layers
        self.conv_input_shape = conv_input_shape
        self.conv_layers = conv_layers
        self.style_input = style_input
        self.style_layers = style_layers
        self.view_layers = view_layers
        self.dense_activation = dense_activation
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer

        ##dynamic input shape
        self.input = [gan_input.input_layer]
        if style_input is not None and style_input is not gan_input:
            self.input.append(style_input.input_layer)
        
        print("GENERATOR TAKES %d INPUTS" % len(self.input))
        self.tracked_layers = {}
        self.viewing_layers = []

    def build(self):
        print("BUILDING GENERATOR DENSE")
        DM = DenseModel(self.gan_input.input_layer,
                        self.dense_layers,
                        self.dense_activation)
        DM_og = DM.build()
        DM_out = Dense(np.prod(self.conv_input_shape))(DM_og)
        DM_out = Reshape(self.conv_input_shape)(DM_out)
        
        print("BUILDING GENERATOR STYLE")
        SM = DenseModel(self.style_input.input_layer,
                        self.style_layers,
                        self.dense_activation)
        SM_out = SM.build()
        print("BUILDING GENERATOR CONV")

        view_channels = self.conv_layers[-1].filters if self.view_layers else None
        CM = ConvolutionalModel(DM_out,
                                self.conv_layers,
                                SM_out,
                                view_channels,
                                self.kernel_regularizer,
                                self.kernel_initializer)
        self.model = CM.build()
        self.tracked_layers = CM.tracked_layers
        self.viewing_layers = CM.viewing_layers
        return self.model

    def get_training_batch(self,batch_size):
        b = [self.gan_input.get_training_batch(batch_size)]
        if self.style_input is not None and self.style_input is not self.gan_input:
            b.append(self.style_input.get_training_batch(batch_size))
        return b

    def get_validation_batch(self,batch_size):
        b = [self.gan_input.get_validation_batch(batch_size)]
        if self.style_input is not None and self.style_input is not self.gan_input:
            b.append(self.style_input.get_validation_batch(batch_size))
        return b
