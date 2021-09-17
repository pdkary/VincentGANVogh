from models.Discriminator import Discriminator
from config.GanConfig import DiscConvLayerConfig,DiscDenseLayerConfig, ActivationConfig, NoneCallable, NormalizationConfig, RegularizationConfig
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss

def get_vgg19(input_channels:int,
              conv_activation:ActivationConfig,
              dense_activation:ActivationConfig,
              final_activation:ActivationConfig,
              normalization:NormalizationConfig,
              optimizer: Optimizer,
              loss_function: Loss,
              kernel_regularizer: RegularizationConfig = NoneCallable,
              kernel_initializer: str = "glorot_uniform",
              output_dim: int = 1,
              minibatch_size:int = 32,
              dropout_rate:float = 0.5,
              lite:bool = False):
    d_c = lambda f,c: DiscConvLayerConfig(f,c,3,conv_activation,normalization)
    d_d = lambda s : DiscDenseLayerConfig(s,dense_activation,dropout_rate)
    d_out = DiscDenseLayerConfig(output_dim, final_activation, 0.0)
    
    D = Discriminator(
        img_shape = (256,256,input_channels),
        disc_conv_layers = [d_c(64,2),d_c(128,2),d_c(256,3),d_c(512,3),d_c(512,3)],
        disc_dense_layers = [d_d(4096),d_d(4096),d_d(1000),d_out],
        minibatch_size = minibatch_size,
        disc_optimizer = optimizer,
        loss_function = loss_function,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer)
    
    if lite:
        D.disc_conv_layers = D.disc_conv_layers[:-1] + 2*[D.disc_conv_layers[-1]]
    return D
    
    
    
