from models.NoiseModel import NoiseModelBase
from models.StyleModel import StyleModelBase
from typing import List, Tuple
from layers.GanInput import GanInput
from tensorflow.keras.optimizers import Optimizer
from config.CallableConfig import ActivationConfig, NormalizationConfig
from tensorflow.keras.losses import Loss

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
               gen_layers: List[GenLayerConfig],
               gen_optimizer: Optimizer,
               loss_function: Loss,
               style_model: StyleModelBase = None,
               noise_model: NoiseModelBase = None,
               normalization: NormalizationConfig = None):
    self.img_shape = img_shape
    self.input_model = input_model
    self.gen_layers = gen_layers,
    self.gen_optimizer = gen_optimizer
    self.loss_function = loss_function
    self.style_model = style_model
    self.noise_model = noise_model
    self.normalization = normalization