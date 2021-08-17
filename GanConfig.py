from typing import Any, Callable, Dict, Tuple,List
from keras.layers import Activation
from keras.engine.base_layer import Layer
from keras.models import Functional
from keras.optimizers import Optimizer

class CallableConfig():
  def __init__(self,
               callable: Callable,
               args: Dict,
               kwargs: Dict = {}):
    self.callable = callable
    self.args = args
    self.kwargs = kwargs
  
  def get(self):
    return self.callable(**self.args,**self.kwargs)

class ActivationConfig(CallableConfig):
  pass

class NormalizationConfig(CallableConfig):
  pass

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
               noise_kernel_size: int,
               gauss_factor:float):
    self.noise_image_size = noise_image_size
    self.noise_kernel_size = noise_kernel_size
    self.gauss_factor = gauss_factor

class GenLayerConfig():
  def __init__(self,
               filters:int,
               convolutions:int,
               kernel_size: int,
               activation: ActivationConfig,
               upsampling = True,
               style = True,
               noise = True):
    self.filters = filters
    self.convolutions = convolutions
    self.kernel_size = kernel_size
    self.activation = activation
    self.upsampling = upsampling
    self.style = style
    self.noise = noise
    
class GeneratorModelConfig():
  def __init__(self,
               img_shape: Tuple[int,int,int],
               gen_constant_shape: Tuple[int,int,int],
               gen_layers: List[GenLayerConfig],
               non_style_normalization: NormalizationConfig,
               gen_loss_function: str,
               gen_optimizer: Optimizer):
    self.img_shape = img_shape
    self.gen_constant_shape = gen_constant_shape
    self.gen_layers = gen_layers,
    self.non_style_normalization = non_style_normalization
    self.gen_loss_function = gen_loss_function
    self.gen_optimizer = gen_optimizer
    
class DiscConvLayerConfig():
  def __init__(self,
               filters: int,
               convolutions: int,
               kernel_size: int,
               activation_config: ActivationConfig,
               normalization: Functional):
    self.filters = filters
    self.convolutions = convolutions
    self.kernel_size = kernel_size
    self.activation_config = activation_config
    self.normalization = normalization

class DiscDenseLayerConfig():
  def __init__(self,
               size: int,
               activation_config: ActivationConfig,
               dropout_rate: int):
    self.size = size
    self.activation_config = activation_config
    self.dropout_rate = dropout_rate

class DiscriminatorModelConfig():
  def __init__(self,
               img_shape: Tuple[int,int,int],
               disc_conv_layers: List[DiscConvLayerConfig],
               disc_dense_layers: List[DiscDenseLayerConfig],
               minibatch,
               minibatch_size,
               disc_loss_function,
               disc_optimizer):
    self.img_shape = img_shape
    self.disc_conv_layers = disc_conv_layers
    self.disc_dense_layers = disc_dense_layers
    self.minibatch = minibatch
    self.minibatch_size = minibatch_size
    self.disc_loss_function = disc_loss_function
    self.disc_optimizer = disc_optimizer

class GanTrainingConfig():
  def __init__(self,
               batch_size,
               preview_rows,
               preview_cols,
               data_path,
               image_type,
               model_name,
               flip_lr,
               load_n_percent):
    self.batch_size = batch_size
    self.preview_rows = preview_rows
    self.preview_cols = preview_cols
    self.data_path = data_path
    self.image_type = image_type
    self.model_name = model_name
    self.flip_lr = flip_lr
    self.load_n_percent = load_n_percent
