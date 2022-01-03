from enum import Enum
from config.DNA import GanDNA, GeneratorDNA

from config.GanConfig import ConvLayerConfig, DenseLayerConfig, SearchableEnum, SimpleActivations
from inputs.GanInput import GanInput
from models.GANBase import GANBase

from models.builders.ConvolutionalModelBuilder import ConvolutionalModelBuilder
from models.builders.DenseModelBuilder import DenseModelBuilder

class Generator(GANBase):
    def build(self):
        ## Dense model
        DM_builder = DenseModelBuilder(self.gan_input.input_layer)
        for d in self.dense_layers:
            DM_builder = DM_builder.block(d)
        ##reshape
        DM_out = DM_builder.reshape(self.conv_input_shape).build()
        ## Convolutional model       
        view_channels = self.conv_layers[-1].filters if self.view_layers else None
        CM_builder = ConvolutionalModelBuilder(
                                input_layer=DM_out,
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
    
    def to_DNA(self, activation_set: SearchableEnum = SimpleActivations):
        dna = super().to_DNA(activation_set=activation_set)
        dna.__class__ = GeneratorDNA
        return dna
    
    @staticmethod
    def from_DNA(gan_input: GanInput, dna: GanDNA):
        gan_base = GANBase.from_DNA(gan_input, dna)
        gan_base.__class__ = Generator
        return gan_base
