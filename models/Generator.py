from typing import List, Tuple

from config.GanConfig import GenLayerConfig, GeneratorDNA
from inputs.GanInput import GanInput
from layers.CallableConfig import ActivationConfig, NoneCallable, RegularizationConfig
from inputs.BatchedInputModel import BatchedInputModel

from models.builders.ConvolutionalModelBuilder import ConvolutionalModelBuilder
from models.builders.DenseModelBuilder import DenseModelBuilder

class Generator(BatchedInputModel):
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
        super().__init__(gan_input)
        self.dense_layers = dense_layers
        self.conv_input_shape = conv_input_shape
        self.conv_layers = conv_layers
        self.view_layers = view_layers
        self.std_dims = std_dims
        self.dense_activation = dense_activation
        self.view_activation = view_activation
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer

        self.tracked_layers = {}
        self.viewing_layers = []

    def build(self):
        ## Dense model
        DM_builder = DenseModelBuilder(self.gan_input.input_layer)
        for d in self.dense_layers:
            DM_builder = DM_builder.block(d,self.dense_activation)
        ##reshape
        DM_out = DM_builder.reshape(self.conv_input_shape).build()
        ## Convolutional model       
        view_channels = self.conv_layers[-1].filters if self.view_layers else None
        CM_builder = ConvolutionalModelBuilder(
                                input=DM_out,
                                view_channels=view_channels,
                                view_activation=self.view_activation,
                                std_dims=self.std_dims,
                                kernel_regularizer=self.kernel_regularizer,
                                kernel_initializer=self.kernel_initializer)
        
        for c in self.conv_layers:
            CM_builder = CM_builder.block(c)

        self.model = CM_builder.build()
        self.tracked_layers = CM_builder.tracked_layers
        return self.model
    
    def toDNA(self):
        return GeneratorDNA(self.dense_layers,self.conv_input_shape,
                            [x.filters for x in self.conv_layers],
                            [x.convolutions for x in self.conv_layers],
                            [x.kernel_size for x in self.conv_layers],
                            [x.upsampling for x in self.conv_layers])
