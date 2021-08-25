from typing import Tuple
from models.GanInput import GanInput
from keras.optimizers import Optimizer
from config.CallableConfig import ActivationConfig, NormalizationConfig

class StyleModelConfig():
  def __init__(self,
               style_latent_size: int,
               style_layer_size: int,
               style_layers: int,
               style_activation: ActivationConfig):
    self.style_latent_size = style_latent_size
    self.style_layer_size = style_layer_size
    self.style_layers = style_layers
    self.style_activation = style_activation

class NoiseModelConfig():
  def __init__(self,
               noise_image_size: Tuple[int,int,int],
               kernel_size: int,
               gauss_factor: float):
    self.noise_image_size = noise_image_size
    self.kernel_size = kernel_size
    self.gauss_factor = gauss_factor

class GenLayerConfig():
  def __init__(self,
               filters:int,
               convolutions:int,
               kernel_size: int,
               activation: ActivationConfig,
               strides:Tuple[int,int] = (1,1),
               transpose: bool = False,
               upsampling: bool = False,
               style: bool = False,
               noise: bool = False):
    self.filters = filters
    self.convolutions = convolutions
    self.kernel_size = kernel_size
    self.activation = activation
    self.strides = strides
    self.transpose = transpose
    self.upsampling = upsampling
    self.style = style
    self.noise = noise
    
class GeneratorModelConfig():
  def __init__(self,
               img_shape: Tuple[int,int,int],
               input_model: GanInput,
               gen_layers: list[GenLayerConfig],
               gen_optimizer: Optimizer,
               style_model_config: StyleModelConfig,
               noise_model_config: NoiseModelConfig,
               normalization: NormalizationConfig):
    self.img_shape = img_shape
    self.input_model = input_model
    self.gen_layers = gen_layers,
    self.gen_optimizer = gen_optimizer
    self.style_model_config = style_model_config
    self.noise_model_config = noise_model_config
    self.normalization = normalization