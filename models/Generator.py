from config.DNA import GeneratorDNA

from config.GanConfig import ConvLayerConfig
from inputs.GanInput import GanInput
from models.GANBase import GANBase

from models.builders.ConvolutionalModelBuilder import ConvolutionalModelBuilder
from models.builders.DenseModelBuilder import DenseModelBuilder

class Generator(GANBase):

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
    
    def to_DNA(self):
        conv_activation = self.conv_layers[0].activation
        return GeneratorDNA(self.dense_layers,self.conv_input_shape,
                            [x.filters for x in self.conv_layers],
                            [x.convolutions for x in self.conv_layers],
                            [x.kernel_size for x in self.conv_layers],
                            [x.upsampling for x in self.conv_layers],
                            self.dense_activation,conv_activation,self.final_activation,
                            self.kernel_regularizer,self.kernel_initializer,
                            self.minibatch_size,
                            self.dropout_rate)
    
    @staticmethod
    def from_DNA(gan_input: GanInput,dna: GeneratorDNA):
        conv_data = zip(dna.conv_filters,dna.conv_layers,dna.conv_kernel_sizes,dna.conv_upsampling)
        conv_layers = [ConvLayerConfig(f,c,k,dna.conv_activation,upsampling=u) for f,c,k,u in conv_data]
        return Generator(gan_input,conv_layers,dna.dense_layers,
                         conv_input_shape=dna.conv_input_shape,
                         dense_activation=dna.dense_activation,
                         final_activation=dna.final_activation,
                         kernel_regularizer=dna.kernel_regularizer,
                         kernel_initializer=dna.kernel_initializer,
                         minibatch_size=dna.minibatch_size,
                         dropout_rate=dna.dropout_rate)
