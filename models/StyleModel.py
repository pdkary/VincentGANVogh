from abc import ABC, abstractmethod
from config.CallableConfig import ActivationConfig
from typing import Tuple
from layers.GanInput import RealImageInput
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Input, Dense, Flatten
import tensorflow as tf

class StyleModelBase(ABC):
    def __init__(self,
                 input_shape: Tuple,
                 activation:ActivationConfig,
                 style_layers:int,
                 style_layer_size:int):
        self.input = Input(shape=input_shape,name="style_model_input")
        self.model = self.input
        self.activation = activation
        self.style_layers = style_layers
        self.style_layer_size = style_layer_size
    
    @abstractmethod
    def get_batch(self,batch_size:int):
        pass
        
class StyleLatentModel(StyleModelBase):
    def __init__(self,
                 input_shape: Tuple, 
                 activation: ActivationConfig,
                 style_layers: int, 
                 style_layer_size: int):
        super().__init__(input_shape,activation)
        for i in range(style_layers):
            self.model = Dense(style_layer_size, kernel_regularizer=L2(), kernel_initializer = 'he_normal')(self.model)
            self.model = self.activation.get()(self.model)
    
    def get_batch(self,batch_size:int):
        return tf.random.normal(shape = (batch_size,self.style_layer_size),dtype=tf.float32)

class ImageStyleModel(StyleModelBase):
    def __init__(self,
                 real_image_input: RealImageInput,
                 activation: ActivationConfig,
                 style_layers: int,
                 style_layer_size: int):
        self.image_source = real_image_input
        super().__init__(real_image_input.input_shape,activation,style_layers,style_layer_size)
        self.model = Flatten()(self.model)
        for i in range(style_layers):
            self.model = Dense(style_layer_size, kernel_regularizer=L2(), kernel_initializer='he_normal')(self.model)
            self.model = self.activation.get()(self.model)
        
    def get_batch(self,batch_size):
        self.image_source.get_batch(batch_size)