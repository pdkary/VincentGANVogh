from abc import ABC, abstractmethod

from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.models import Model
from config.GanConfig import ActivationConfig, RegularizationConfig
from typing import Tuple
from layers.GanInput import RealImageInput
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Input, Dense, Flatten
import tensorflow as tf

class StyleModelBase(ABC):
    def __init__(self,
                 input_shape: Tuple,
                 style_layers:int,
                 style_layer_size:int,
                 activation:ActivationConfig,
                 kernel_regularizer:RegularizationConfig,
                 kernel_initializer: str = "glorot_uniform"):
        self.input_shape = input_shape
        self.input = Input(shape=input_shape,name="style_model_input")
        self.activation = activation
        self.style_layers = style_layers
        self.style_layer_size = style_layer_size
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
    
    @abstractmethod
    def get_batch(self,batch_size:int):
        pass
        
class LatentStyleModel(StyleModelBase):
    def __init__(self, 
                 input_shape: Tuple, 
                 style_layers: int, 
                 style_layer_size: int, 
                 activation: ActivationConfig, 
                 kernel_regularizer: RegularizationConfig, 
                 kernel_initializer: str = "glorot_uniform"):
        super().__init__(input_shape, style_layers, style_layer_size, activation, kernel_regularizer, kernel_initializer)
        
        model = self.input
        for i in range(self.style_layers):
            model = Dense(self.style_layer_size, 
                          kernel_regularizer=self.kernel_regularizer.get(),
                          kernel_initializer = self.kernel_initializer)(model)
            model = self.activation.get()(model)
        self.model = model
    
    def get_batch(self,batch_size:int):
        return tf.random.normal(shape = (batch_size,self.input_shape),dtype=tf.float32)

class ImageStyleModel(StyleModelBase):
    def __init__(self,
                 real_image_input: RealImageInput,
                 activation: ActivationConfig,
                 style_layers: int,
                 style_layer_size: int,
                 downsample_factor: int = 1):
        self.image_source = real_image_input
        self.downsample_factor = (downsample_factor,downsample_factor)
        super().__init__(real_image_input.input_shape,activation,style_layers,style_layer_size)
        
        model = MaxPooling2D(self.downsample_factor)(self.input)
        model = Flatten()(model) if self.style_layers > 0 else model
        for i in range(self.style_layers):
            model = Dense(self.style_layer_size, 
                          kernel_regularizer=self.kernel_regularizer.get(), 
                          kernel_initializer=self.kernel_initializer)(model)
            model = self.activation.get()(model)
        self.model = model
        
    def get_batch(self,batch_size):
        return self.image_source.get_batch(batch_size)
