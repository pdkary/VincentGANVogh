from config.DNA import DiscriminatorDNA, GanDNA
from config.GanConfig import SimpleActivations
from helpers.SearchableEnum import SearchableEnum
from inputs.GanInput import GanInput

from models.builders.ConvolutionalModelBuilder import ConvolutionalModelBuilder
from models.builders.DenseModelBuilder import DenseModelBuilder
from models.GANBase import GANBase


class Discriminator(GANBase):
    def build(self):
        # convolutional model
        CM_builder = ConvolutionalModelBuilder(self.input,
                                               std_dims=self.std_dims,
                                               kernel_regularizer=self.kernel_regularizer,
                                               kernel_initializer=self.kernel_initializer)

        for c in self.conv_layers:
            CM_builder = CM_builder.block(c)

        self.conv_out = CM_builder.flatten().build()
        # dense model
        DM_builder = DenseModelBuilder(self.conv_out)
        self.view_layers = CM_builder.view_layers

        for d in self.dense_layers:
            DM_builder = DM_builder.block(d)

        self.dense_out = DM_builder.build()
        self.feature_layers = DM_builder.feature_layers
        return self.dense_out

    @property
    def input_shape(self):
        return self.gan_input.input_shape

    @property
    def output_shape(self):
        return [self.dense_layers[-1].size]
    
    def to_DNA(self, activation_set: SearchableEnum = SimpleActivations):
        dna = super().to_DNA(activation_set=activation_set)
        dna.__class__ = DiscriminatorDNA
        return dna

    @staticmethod
    def from_DNA(gan_input: GanInput, dna: GanDNA):
        gan_base = GANBase.from_DNA(gan_input, dna)
        gan_base.__class__ = Discriminator
        return gan_base
    
