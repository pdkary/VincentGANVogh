
from config.GeneratorConfig import GenLayerConfig, GeneratorModelConfig
from models.GanInput import EncoderInput
from tensorflow.python.keras.losses import BinaryCrossentropy
from config.DiscriminatorConfig import DiscConvLayerConfig, DiscDenseLayerConfig, DiscriminatorModelConfig
from tensorflow.keras.optimizers import Optimizer
from config.CallableConfig import ActivationConfig, NormalizationConfig

def get_encoder(input_channels:int,
                encoded_space_size: int,
                conv_activation:ActivationConfig,
                dense_activation:ActivationConfig,
                final_activation:ActivationConfig,
                normalization:NormalizationConfig,
                optimizer: Optimizer,
                minibatch_size:int = 32,
                dropout_rate:float = 0.5):
    d_c = lambda f,c: DiscConvLayerConfig(f,c,3,conv_activation,normalization)
    d_d = lambda s : DiscDenseLayerConfig(s,dense_activation,dropout_rate)
    d_out = DiscDenseLayerConfig(encoded_space_size,    final_activation, 0.0)
    
    return DiscriminatorModelConfig(
        img_shape = (256,256,input_channels),
        disc_conv_layers = [d_c(64,2),d_c(128,2),d_c(256,3),d_c(512,3),d_c(512,3)],
        disc_dense_layers = [d_d(4096),d_d(4096),d_d(1000),d_out],
        minibatch=True,
        minibatch_size=minibatch_size,
        disc_optimizer = optimizer,
        loss_function=BinaryCrossentropy())
    
def get_decoder(output_channels: int,
                encoded_space_size: int,
                conv_activation: ActivationConfig,
                final_activation: ActivationConfig,
                normalization: NormalizationConfig,
                optimizer: Optimizer):
    gl = lambda f,c: GenLayerConfig(f,c,3,conv_activation,upsampling=True)
    gf = GenLayerConfig(output_channels,1,1,final_activation)
    return GeneratorModelConfig(
        img_shape = (256,256,output_channels),
        input_model = EncoderInput(encoded_space_size),
        style_model_config = None,
        noise_model_config = None,
        gen_layers=[gl(512,3),gl(512,3),gl(256,3),gl(128,3),gl(64,2),gl(32,2),gf],
        normalization=normalization,
        gen_optimizer=optimizer)
    