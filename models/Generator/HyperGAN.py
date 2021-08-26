from models.Discriminator import Discriminator
from models.VGG19 import get_vgg19
from models.Generator.Generator import Generator
from keras.optimizer_v2.adam import Adam
from models.GanInput import GenLatentSpaceInput
from config.GeneratorConfig import GenLayerConfig, GeneratorModelConfig
from third_party_layers.InstanceNormalization import InstanceNormalization
from keras.layers import BatchNormalization, Activation, LeakyReLU
from config.CallableConfig import ActivationConfig, NormalizationConfig
from keras_tuner import HyperModel

class HyperDiscriminator(HyperModel):
    def __init__(self, name, tunable):
        super().__init__(name=name, tunable=tunable)
        
    def build(self, hp):
        convolutional_relu_alpha = hp.Float("conv_relu_alpha", 0.0, 1.0,step=0.03)
        dense_relu_alpha = hp.Float("dense_relu_alpha", 0.0, 1.0, step=0.03)

        leakyRELU_conv = ActivationConfig(LeakyReLU, dict(alpha=convolutional_relu_alpha))
        leakyRELU_dense = ActivationConfig(LeakyReLU, dict(alpha=dense_relu_alpha))
        sigmoid = ActivationConfig(Activation, dict(activation="sigmoid"))

        batch_norm_momentum = hp.Float("batch_norm_momentum", 0.0, 1.0,step=0.05)

        self.norm_dict = {
            "batch_norm": NormalizationConfig(BatchNormalization, dict(momentum=batch_norm_momentum)),
            "instance_norm": NormalizationConfig(InstanceNormalization)
        }
        normalization = hp.Choice("normalization", ["batch_norm", "instance_norm"])
        
        disc_model_config = get_vgg19(leakyRELU_conv, leakyRELU_dense, sigmoid, self.norm_dict[normalization])
        self.D = Discriminator(disc_model_config)
        return self.D.build()
    
class HyperGenerator(HyperModel):
    def __init__(self, name, tunable,batch_size, preview_size):
        super().__init__(name=name, tunable=tunable)
        self.batch_size = batch_size
        self.preview_size = preview_size
    
    def build(self,hp):
        learning_rate = hp.Float("learning_rate", 1e-6, 1e-2)
        beta_1 = hp.Float("beta_1", 0.0, 0.99)
        beta_2 = hp.Float("beta_2", 0.0, 0.99)

        latent_space_size = hp.Int("latent_space_size", 10, 500,step=10)
        kernel_size = hp.Choice("kernel_size", [1, 3, 5, 7])
        convolutions_per_layer = hp.Int("convolutions_per_layer", 1, 5)

        layer_1_filters = hp.Int("layer_1_filters", 1, 512, step=32)
        layer_2_filters = hp.Int("layer_2_filters", 1, 512, step=32)
        layer_3_filters = hp.Int("layer_3_filters", 1, 512, step=32)
        layer_4_filters = hp.Int("layer_4_filters", 1, 512, step=32)
        layer_5_filters = hp.Int("layer_5_filters", 1, 512, step=32)
        layer_6_filters = hp.Int("layer_6_filters", 1, 512, step=32)

        convolutional_relu_alpha = hp.Float("conv_relu_alpha", 0.0, 1.0,step=0.03)
        dense_relu_alpha = hp.Float("dense_relu_alpha", 0.0, 1.0, step=0.03)

        leakyRELU_conv = ActivationConfig(LeakyReLU, dict(alpha=convolutional_relu_alpha))
        leakyRELU_dense = ActivationConfig(LeakyReLU, dict(alpha=dense_relu_alpha))
        sigmoid = ActivationConfig(Activation, dict(activation="sigmoid"))

        batch_norm_momentum = hp.Float("batch_norm_momentum", 0.0, 1.0,step=0.05)

        self.norm_dict = {
            "batch_norm": NormalizationConfig(BatchNormalization, dict(momentum=batch_norm_momentum)),
            "instance_norm": NormalizationConfig(InstanceNormalization)
        }
        normalization = hp.Choice("normalization", ["batch_norm", "instance_norm"])

        image_shape = (256, 256, 3)

        gen_model_config = GeneratorModelConfig(
            img_shape=image_shape,
            input_model=GenLatentSpaceInput(latent_space_size, (4, 4, 512), 100, 0, leakyRELU_dense),
            gen_layers=[
                GenLayerConfig(layer_1_filters,  convolutions_per_layer, kernel_size, leakyRELU_conv, upsampling=True),
                GenLayerConfig(layer_2_filters,  convolutions_per_layer, kernel_size, leakyRELU_conv, upsampling=True),
                GenLayerConfig(layer_3_filters,  convolutions_per_layer, kernel_size, leakyRELU_conv, upsampling=True),
                GenLayerConfig(layer_4_filters,  convolutions_per_layer, kernel_size, leakyRELU_conv, upsampling=True),
                GenLayerConfig(layer_5_filters,  convolutions_per_layer, kernel_size, leakyRELU_conv, upsampling=True),
                GenLayerConfig(layer_6_filters,  convolutions_per_layer, kernel_size, leakyRELU_conv, upsampling=True),
                GenLayerConfig(3,    1, 1, sigmoid,   upsampling=False)],
            gen_optimizer=Adam(learning_rate=learning_rate, beta_1=beta_1,beta_2=beta_2, epsilon=1e-7),
            normalization=self.norm_dict[normalization]
        )
        self.G = Generator(gen_model_config,self.batch_size,self.preview_size)
        return self.G.build_generator()