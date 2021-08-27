from tensorflow.keras.models import Model
from config.DiscriminatorConfig import DiscConvLayerConfig, DiscDenseLayerConfig, DiscriminatorModelConfig
from models.Discriminator import Discriminator
from models.VGG19 import get_vgg19
from models.Generator.Generator import Generator
from tensorflow.keras.optimizers import Adam
from models.GanInput import GenLatentSpaceInput
from config.GeneratorConfig import GenLayerConfig, GeneratorModelConfig
from third_party_layers.InstanceNormalization import InstanceNormalization
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU
from config.CallableConfig import ActivationConfig, NormalizationConfig
from keras_tuner import HyperModel

leakyRELU_conv = ActivationConfig(LeakyReLU, dict(alpha=0.05))
leakyRELU_dense = ActivationConfig(LeakyReLU, dict(alpha=0.1))
sigmoid = ActivationConfig(Activation, dict(activation="sigmoid"))
instance_norm = NormalizationConfig(InstanceNormalization)
    
class HyperGAN(HyperModel):
    def __init__(self, name, tunable,batch_size, preview_size):
        super().__init__(name=name, tunable=tunable)
        self.batch_size = batch_size
        self.preview_size = preview_size
        
        self.disc_model_config = DiscriminatorModelConfig(
                img_shape = (64,64,3),
                disc_conv_layers=[
                    DiscConvLayerConfig(64,  2, 3, sigmoid, instance_norm),
                    DiscConvLayerConfig(128, 2, 3, sigmoid, instance_norm),
                    DiscConvLayerConfig(256, 3, 3, sigmoid, instance_norm),
                    DiscConvLayerConfig(512, 3, 3, sigmoid, instance_norm)],
                disc_dense_layers=[
                    DiscDenseLayerConfig(4096, leakyRELU_dense, 0.5),
                    DiscDenseLayerConfig(4096, leakyRELU_dense, 0.5),
                    DiscDenseLayerConfig(1000, leakyRELU_dense, 0.5),
                    DiscDenseLayerConfig(1,    sigmoid, 0.5)],
                minibatch=True,
                minibatch_size=32,
                disc_optimizer = Adam(learning_rate=0.002,beta_1=0.0,beta_2=0.99,epsilon=1e-8)
        )
        
        self.D = Discriminator(self.disc_model_config)
        self.discriminator = self.D.build()
    
    def build(self,hp):
        kernel_size = hp.Choice("kernel_size", [1, 3, 5, 7])
        convolutions_per_layer = hp.Int("convolutions_per_layer", 1, 5)
        
        layer_1_filters = hp.Int("layer_1_filters", 1, 512, step=32)
        layer_2_filters = hp.Int("layer_2_filters", 1, 512, step=32)
        layer_3_filters = hp.Int("layer_3_filters", 1, 512, step=32)
        layer_4_filters = hp.Int("layer_4_filters", 1, 512, step=32)

        convolutional_relu_alpha = hp.Float("conv_relu_alpha", 0.0, 1.0,step=0.03)

        lr_conv = ActivationConfig(LeakyReLU, dict(alpha=convolutional_relu_alpha))

        batch_norm_momentum = hp.Float("batch_norm_momentum", 0.0, 1.0,step=0.05)

        self.norm_dict = {
            "batch_norm": NormalizationConfig(BatchNormalization, dict(momentum=batch_norm_momentum)),
            "instance_norm": NormalizationConfig(InstanceNormalization)
        }
        normalization = hp.Choice("normalization", ["batch_norm", "instance_norm"])

        gen_model_config = GeneratorModelConfig(
            img_shape=(64,64,3),
            input_model=GenLatentSpaceInput(100, (4, 4, 512), 100, 0, leakyRELU_dense),
            gen_layers=[
                GenLayerConfig(layer_1_filters,  convolutions_per_layer, kernel_size, lr_conv, upsampling=True),
                GenLayerConfig(layer_2_filters,  convolutions_per_layer, kernel_size, lr_conv, upsampling=True),
                GenLayerConfig(layer_3_filters,  convolutions_per_layer, kernel_size, lr_conv, upsampling=True),
                GenLayerConfig(layer_4_filters,  convolutions_per_layer, kernel_size, lr_conv, upsampling=True),
                GenLayerConfig(3,    1, 1, sigmoid,   upsampling=False)],
            gen_optimizer=Adam(learning_rate=0.002, beta_1=0.0,beta_2=0.99, epsilon=1e-7),
            normalization=self.norm_dict[normalization]
        )
        
        self.G = Generator(gen_model_config,self.batch_size,self.preview_size)
        self.generator = self.G.build(print_summary=False)
        
        D = self.discriminator
        D.trainable = False
        out = D(self.G.functional_model)
        gen_model = Model(inputs=self.G.input,outputs=out,name="Generator")
        gen_model.compile(optimizer=self.G.gen_optimizer,
                           loss="binary_crossentropy",
                           metrics=['accuracy'])
        return gen_model
        