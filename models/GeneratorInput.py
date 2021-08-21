from typing import Tuple
from keras.backend import prod
from keras.layers import Input, Dense
from keras.layers.core import Activation, Reshape
import tensorflow as tf
import numpy as np
from abc import ABC

class GeneratorInput(ABC):
    def __init__(self,input_shape,name="generator_input"):
        self.input_shape = input_shape
        self.input = Input(shape=input_shape,dtype=tf.float32,name=name)
        
    def get_batch(self,batch_size):
        pass

class GenConstantInput(GeneratorInput):
    def __init__(self, input_shape: Tuple):
        super().__init__(input_shape,name="gen_constant_input")
        self.constant = tf.constant(tf.random.normal(shape=input_shape,dtype=tf.float32))
        self.model = Activation('linear')(self.input)
    
    def get_batch(self, batch_size):
        gc_batch = np.full((batch_size,*self.input_shape),0.0,dtype=np.float32)
        for i in range(batch_size):
            gc_batch[i] = self.constant
        return gc_batch
        
class GenLatentSpaceInput(GeneratorInput):
    def __init__(self, input_shape: int,output_shape:Tuple,layer_size,layers):
        super().__init__(input_shape,name="gen_latent_space_input")
        self.model = self.input
        for i in range(layers):
            self.model = Dense(layer_size)(self.model)
        self.model = Dense(prod(output_shape))(self.model)
        self.model = Reshape(output_shape)(self.model)
    
    def get_batch(self, batch_size):
        return tf.random.normal(shape=(batch_size,self.input_shape))