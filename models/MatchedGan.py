from models.Generator import Generator
from layers.GanInput import GanInput, GenLatentSpaceInput
from layers.CallableConfig import ActivationConfig, RegularizationConfig
from models.NoiseModel import NoiseModelBase
from models.StyleModel import StyleModelBase
from typing import List, Tuple
from models.Discriminator import Discriminator
from config.GanConfig import DiscConvLayerConfig,DiscDenseLayerConfig, GenLayerConfig
from layers.CallableConfig import ActivationConfig, NoneCallable, NormalizationConfig, RegularizationConfig
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.losses import Loss

def get_matched_gan(img_shape:Tuple[int,int,int],
                    gen_input_model: GanInput,
                    layer_sizes:List[Tuple],
                    disc_optimizer:Optimizer,
                    disc_loss_func: Loss,
                    disc_conv_activation: ActivationConfig,
                    disc_dense_activation: ActivationConfig,
                    disc_final_activation: ActivationConfig,
                    gen_optimizer:Optimizer,
                    gen_loss_func: Loss,
                    gen_conv_activation: ActivationConfig,
                    gen_dense_activation: ActivationConfig,
                    gen_final_activation,
                    disc_output_dim: int = 1,
                    minibatch_size: int = 32,
                    dropout_rate: float = 0.5,
                    style_model: StyleModelBase = None,
                    noise_model: NoiseModelBase = None,
                    normalization: NormalizationConfig = NoneCallable,
                    kernel_regularizer: RegularizationConfig = NoneCallable,
                    kernel_initializer: str = "glorot_uniform"):
    dcl = lambda f,c: DiscConvLayerConfig(f,c,3,dropout_rate,disc_conv_activation,normalization)
    ddl = lambda s: DiscDenseLayerConfig(s,disc_dense_activation,dropout_rate)
    dout = DiscDenseLayerConfig(disc_output_dim,disc_final_activation,0.0)
    

    gen_input_model = GenLatentSpaceInput(100,(2,2,1024),512,3,gen_dense_activation)
    gl = lambda f,c: GenLayerConfig(f,c,3,gen_conv_activation,upsampling=True,style=True,noise=True)
    gen_output_layer = GenLayerConfig(img_shape[-1],1,1,gen_final_activation,style=True)
    
    disc_conv_layers = [dcl(f,c) for f,c in layer_sizes]
    disc_dense_layers = [ddl(4096),ddl(4096),ddl(1000),dout]
    
    gen_layers = [gl(f,c) for f,c in reversed(layer_sizes)]
    gen_layers.append(gen_output_layer)
    
    G = Generator(img_shape,gen_input_model,gen_layers,gen_optimizer,
                  gen_loss_func,style_model,noise_model,normalization,
                  kernel_regularizer,kernel_initializer)
    D = Discriminator(img_shape,disc_conv_layers,disc_dense_layers,
                      minibatch_size,disc_optimizer,disc_loss_func,
                      kernel_regularizer,kernel_initializer)
    
    return D,G
