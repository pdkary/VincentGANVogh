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
        view_channels = self.gan_input.input_shape[-1] if self.view_layers else None
        
        CM_builder = ConvolutionalModelBuilder(self.input,
                                               view_channels=view_channels,
                                               view_activation=self.conv_layers[-1].activation,
                                               std_dims=self.std_dims,
                                               kernel_regularizer=self.kernel_regularizer,
                                               kernel_initializer=self.kernel_initializer)

        for c in self.conv_layers:
            CM_builder = CM_builder.block(c)

        self.conv_out = CM_builder.flatten().build()
        self.tracked_layers = CM_builder.tracked_layers
        # dense model
        DM_builder = DenseModelBuilder(self.conv_out)

        for d in self.dense_layers:
            DM_builder = DM_builder.block(d)

        self.dense_out = DM_builder.build()
        return self.dense_out
    
    def to_DNA(self, activation_set: SearchableEnum = SimpleActivations):
        dna = super().to_DNA(activation_set=activation_set)
        dna.__class__ = DiscriminatorDNA
        return dna

    @staticmethod
    def from_DNA(gan_input: GanInput, dna: GanDNA):
        gan_base = GANBase.from_DNA(gan_input, dna)
        gan_base.__class__ = Discriminator
        return gan_base
    
