from helpers.DataHelper import DataHelper
from config.TrainingConfig import DataConfig
from typing import Tuple
from keras.backend import prod
from keras.layers import Input, Dense
from keras.layers.core import Activation, Reshape
import tensorflow as tf
import numpy as np
from abc import ABC

class GanInput(ABC):
    def __init__(self,input_shape,name="gan_input"):
        self.input_shape = input_shape
        self.input = Input(shape=input_shape,dtype=tf.float32,name=name)
        self.model = Activation('linear')(self.input)
        
    def get_batch(self,batch_size,batches=1):
        pass

class GenConstantInput(GanInput):
    def __init__(self, input_shape: Tuple):
        super().__init__(input_shape,name="gen_constant_input")
        self.constant = tf.constant(tf.random.normal(shape=input_shape,dtype=tf.float32))
        self.model = Activation('linear')(self.input)
    
    def get_batch(self, batch_size,batches=1):
        gc_batch = np.full((batch_size,*self.input_shape),0.0,dtype=np.float32)
        for i in range(batch_size):
            gc_batch[i] = self.constant
        return gc_batch
        
class GenLatentSpaceInput(GanInput):
    def __init__(self, input_shape: int,output_shape:Tuple,layer_size,layers):
        super().__init__(input_shape,name="gen_latent_space_input")
        self.model = self.input
        for i in range(layers):
            self.model = Dense(layer_size)(self.model)
        self.model = Dense(prod(output_shape))(self.model)
        self.model = Reshape(output_shape)(self.model)
    
    def get_batch(self, batch_size, batches=1):
        return tf.random.normal(shape=(batch_size,self.input_shape))

class RealImageInput(GanInput):
    def __init__(self,data_config: DataConfig):
        super().__init__(data_config.image_shape,name="real_image_input")
        self.data_helper = DataHelper(data_config)
        self.preview_size = data_config.preview_cols*data_config.preview_rows
    
    def load(self):
        print("Preparing Dataset".upper())
        self.images = self.data_helper.load_data()
        self.dataset = tf.data.Dataset.from_tensor_slices(self.images)
        self.dataset_size = len(self.images)
        print("DATASET LOADED")
    
    def save(self,epoch,images):
        self.data_helper.save_images(epoch,images)
    
    def get_batch(self, batch_size, batches=1):
        self.dataset = self.dataset.shuffle(self.dataset_size)
        return self.dataset.take(batch_size*batches).batch(batch_size)
   
        