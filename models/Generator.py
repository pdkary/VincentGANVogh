from typing import List, Tuple

from tensorflow.keras.layers import Dense, Reshape
from config.GanConfig import GenLayerConfig
from inputs.GanInput import GanInput
from layers.CallableConfig import ActivationConfig, NoneCallable, RegularizationConfig

from models.ConvolutionalModel import ConvolutionalModelBuilder
from models.DenseModel import DenseModelBuilder
import numpy as np

class Generator():
    def __init__(self,
                 gan_input: GanInput,
                 dense_layers: List[int],
                 conv_input_shape: Tuple[int],
                 conv_layers: List[GenLayerConfig],
                 view_layers: bool = False,
                 std_dims: List[int] = [1,2,3],
                 dense_activation: ActivationConfig = NoneCallable,
                 view_activation: ActivationConfig = NoneCallable,
                 kernel_regularizer:RegularizationConfig = NoneCallable,
                 kernel_initializer:str = "glorot_uniform"):
        self.gan_input = gan_input
        self.dense_layers = dense_layers
        self.conv_input_shape = conv_input_shape
        self.conv_layers = conv_layers
        self.view_layers = view_layers
        self.std_dims = std_dims
        self.dense_activation = dense_activation
        self.view_activation = view_activation
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer

        self.input = [gan_input.input_layer]
        
        print("GENERATOR TAKES %d INPUTS" % len(self.input))
        self.tracked_layers = {}
        self.viewing_layers = []

    def build(self):
        print("BUILDING GENERATOR DENSE")
        DM_builder = DenseModelBuilder(self.gan_input.input_layer)
        for d in self.dense_layers:
            DM_builder = DM_builder.dense_layer(d,self.dense_activation)
        DM_og = DM_builder.build()

        DM_out = Dense(np.prod(self.conv_input_shape))(DM_og)
        DM_out = Reshape(self.conv_input_shape)(DM_out)
        
        print("BUILDING GENERATOR CONV")
        view_channels = self.conv_layers[-1].filters if self.view_layers else None
        CM_builder = ConvolutionalModelBuilder(
                                input=DM_out,
                                view_channels=view_channels,
                                view_activation=self.view_activation,
                                std_dims=self.std_dims,
                                kernel_regularizer=self.kernel_regularizer,
                                kernel_initializer=self.kernel_initializer)
        
        for c in self.conv_layers:
            CM_builder = CM_builder.conv_block(c)

        self.model = CM_builder.build()
        self.tracked_layers = CM_builder.tracked_layers
        return self.model

    def get_training_batch(self,batch_size):
        b = [self.gan_input.get_training_batch(batch_size)]
        return b

    def get_validation_batch(self,batch_size):
        b = [self.gan_input.get_validation_batch(batch_size)]
        return b
