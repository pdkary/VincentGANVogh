from typing import List

from config.GanConfig import GenLayerConfig
from inputs.GanInput import GanInput
from layers.CallableConfig import NoneCallable, RegularizationConfig

from models.ConvolutionalModel import ConvolutionalModel


class Generator():
    def __init__(self,
                 gan_input: GanInput,
                 gen_layers: List[GenLayerConfig],
                 kernel_regularizer:RegularizationConfig = NoneCallable,
                 kernel_initializer:str = "glorot_uniform"):
        self.gan_input = gan_input
        self.input = gan_input.input_layer
        self.gen_layers = gen_layers
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.tracked_layers = {}

    def build(self):
        CM = ConvolutionalModel(self.input,self.gen_layers,
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
