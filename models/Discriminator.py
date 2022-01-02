from copy import deepcopy
from typing import List

from config.GanConfig import DiscConvLayerConfig
from inputs.GanInput import RealImageInput
from layers.CallableConfig import (ActivationConfig, NoneCallable,
                                   RegularizationConfig)
from inputs.BatchedInputModel import BatchedInputModel

from models.builders.ConvolutionalModelBuilder import ConvolutionalModelBuilder
from models.builders.DenseModelBuilder import DenseModelBuilder
from models.Generator import Generator


class Discriminator(BatchedInputModel):
    def __init__(self,
                 real_image_input: RealImageInput,
                 conv_layers: List[DiscConvLayerConfig],
                 dense_layers: List[int],
                 minibatch_size: int = 0,
                 dropout_rate: float = 0.0,
                 view_layers: bool = False,
                 std_dims: List[int] = [1,2],
                 dense_activation: ActivationConfig = NoneCallable,
                 final_activation: ActivationConfig = NoneCallable,
                 kernel_regularizer: RegularizationConfig = NoneCallable,
                 kernel_initializer: str = "glorot_uniform"):
        super().__init__(real_image_input)
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.minibatch_size = minibatch_size
        self.dropout_rate = dropout_rate
        self.view_layers = view_layers
        self.std_dims = std_dims
        self.dense_activation = dense_activation
        self.final_activation = final_activation
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.viewing_layers = []
    
    def build(self):
        ##convolutional model
        view_channels = self.gan_input.input_shape[-1] if self.view_layers else None
        CM_builder = ConvolutionalModelBuilder(self.input,
                                view_channels=view_channels,
                                view_activation=self.final_activation,
                                std_dims=self.std_dims,
                                kernel_regularizer=self.kernel_regularizer,
                                kernel_initializer=self.kernel_initializer)

        for c in self.conv_layers:
            CM_builder = CM_builder.block(c)
        
        self.conv_out = CM_builder.flatten().build()
        ##dense model
        DM_builder = DenseModelBuilder(self.conv_out)

        for d in self.dense_layers:
            DM_builder = DM_builder.block(d,self.dense_activation,self.dropout_rate,self.minibatch_size)
        
        self.dense_out = self.final_activation.get()(DM_builder.build())
        self.tracked_layers = CM_builder.tracked_layers
        return self.dense_out
    
    ## Alternative initializer that reverses a generator
    @staticmethod
    def from_generator(generator:Generator,
                       real_image_input: RealImageInput,
                       final_activation: ActivationConfig,
                       output_dim: int = None,
                       minibatch_size: int = 0,
                       dropout_rate: float = 0.0,
                       viewable: bool = False):
        conv_layers = [x.flip() for x in reversed(deepcopy(generator.conv_layers))]
        
        dense_layers = list(reversed(deepcopy(generator.dense_layers)))
        dense_layers.append(generator.gan_input.input_shape[-1])
        if output_dim is not None:
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
                             std_dims=generator.std_dims,
                             kernel_regularizer=kr,
                             kernel_initializer=ki)
