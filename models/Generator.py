from typing import List, Tuple

from config.GanConfig import GenLayerConfig
from inputs.GanInput import GanInput
from layers.CallableConfig import ActivationConfig, NoneCallable, RegularizationConfig

from models.ConvolutionalModel import ConvolutionalModel
from models.DenseModel import DenseModel


class Generator():
    def __init__(self,
                 gan_input: GanInput,
                 dense_layers: List[int],
                 conv_input_shape: Tuple[int],
                 conv_layers: List[GenLayerConfig],
                 dense_activation: ActivationConfig = NoneCallable,
                 kernel_regularizer:RegularizationConfig = NoneCallable,
                 kernel_initializer:str = "glorot_uniform"):
        self.gan_input = gan_input
        self.dense_layers = dense_layers
        self.conv_input_shape = conv_input_shape
        self.conv_layers = conv_layers
        self.dense_activation = dense_activation
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer

        self.input = gan_input.input_layer
        self.tracked_layers = {}

    def build(self):
        DM = DenseModel(self.input,self.dense_layers,self.dense_activation)
        DM_out = DM.build()
        CM = ConvolutionalModel(DM_out,self.conv_layers,
                                self.kernel_regularizer,
                                self.kernel_initializer)
        self.model = CM.build()
        self.tracked_layers = CM.tracked_layers
        return self.model

    def get_training_batch(self,batch_size):
        b = [self.gan_input.get_training_batch(batch_size)]
        return b

    def get_validation_batch(self,batch_size):
        b = [self.gan_input.get_validation_batch(batch_size)]
        return b
