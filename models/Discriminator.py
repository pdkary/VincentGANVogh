from config.DNA import DiscriminatorDNA
from config.GanConfig import ConvLayerConfig
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
                                               view_activation=self.final_activation,
                                               std_dims=self.std_dims,
                                               kernel_regularizer=self.kernel_regularizer,
                                               kernel_initializer=self.kernel_initializer)

        for c in self.conv_layers:
            CM_builder = CM_builder.block(c)

        self.conv_out = CM_builder.flatten().build()
        # dense model
        DM_builder = DenseModelBuilder(self.conv_out)

        for d in self.dense_layers:
            DM_builder = DM_builder.block(
                d, self.dense_activation, self.dropout_rate, self.minibatch_size)

        self.dense_out = self.final_activation.get()(DM_builder.build())
        self.tracked_layers = CM_builder.tracked_layers
        return self.dense_out

    def to_DNA(self):
        c_act = self.conv_layers[0].activation
        return DiscriminatorDNA(self.dense_layers, self.gan_input.input_shape,
                                [x.filters for x in self.conv_layers],
                                [x.convolutions for x in self.conv_layers],
                                [x.kernel_size for x in self.conv_layers],
                                [x.downsampling for x in self.conv_layers],
                                self.dense_activation, c_act,
                                self.final_activation,
                                self.kernel_regularizer,
                                self.kernel_initializer,
                                self.minibatch_size,
                                self.dropout_rate)

    @staticmethod
    def from_DNA(gan_input: GanInput, dna: DiscriminatorDNA):
        conv_data = zip(dna.conv_filters, dna.conv_layers,
                        dna.conv_kernel_sizes, dna.conv_upsampling)
        conv_layers = [ConvLayerConfig(
            f, c, k, dna.conv_activation, downsampling=u) for f, c, k, u in conv_data]
        return Discriminator(gan_input, conv_layers, dna.dense_layers,
                             conv_input_shape=gan_input.input_shape,
                             minibatch_size=dna.minibatch_size,
                             dropout_rate=dna.dropout_rate,
                             dense_activation=dna.dense_activation,
                             final_activation=dna.final_activation,
                             kernel_regularizer=dna.kernel_regularizer,
                             kernel_initializer=dna.kernel_initializer)
