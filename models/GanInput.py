from config.CallableConfig import ActivationConfig
from tensorflow.keras.regularizers import L2
from helpers.DataHelper import DataHelper
from config.TrainingConfig import DataConfig
from typing import Tuple
from tensorflow.keras.backend import prod
from tensorflow.keras.layers import Input, Dense, Activation, Reshape
import tensorflow as tf
import numpy as np
from abc import ABC

class GanInput(ABC):
    def __init__(self,input_shape: Tuple[int,int,int],name="gan_input"):
        self.input_shape = input_shape
        self.input = Input(shape=input_shape,dtype=tf.float32,name=name)
        self.model = Activation('linear')(self.input)
        
    def get_batch(self,batch_size):
        pass

class EncodedInput(GanInput):
    def __init__(self,input_shape: int,buffer_size:int = 1000):
        super().__init__((input_shape,),name="encoded_input")
        self.buffer_size = buffer_size
        self.dataset = []
        self.looking = 0
    
    def update(self,encoded_val):
        self.dataset.append(encoded_val)
        if len(self.dataset) > self.buffer_size:
            self.dataset = self.dataset[1:]
    
    def get_batch(self, batch_size):
        out = self.dataset[self.looking:self.looking+batch_size]
        self.looking+=batch_size
        return out
    
class GenConstantInput(GanInput):
    def __init__(self, input_shape: Tuple):
        super().__init__(input_shape,name="gen_constant_input")
        self.constant = tf.constant(tf.random.normal(shape=input_shape,dtype=tf.float32))
        self.model = Activation('linear')(self.input)
    
    def get_batch(self, batch_size):
        gc_batch = np.full((batch_size,*self.input_shape),0.0,dtype=np.float32)
        for i in range(batch_size):
            gc_batch[i] = self.constant
        return gc_batch
        
class GenLatentSpaceInput(GanInput):
    def __init__(self, input_shape: int,output_shape:Tuple,layer_size,layers, activation: ActivationConfig):
        super().__init__(input_shape,name="gen_latent_space_input")
        self.model = self.input
        for i in range(layers):
            self.model = Dense(layer_size,kernel_initializer="he_normal",kernel_regularizer=L2())(self.model)
            self.model = activation.get()(self.model)
        self.model = Dense(prod(output_shape),kernel_initializer="he_normal",kernel_regularizer=L2())(self.model)
        self.model = activation.get()(self.model)
        self.model = Reshape(output_shape)(self.model)
    
    def get_batch(self, batch_size):
        return tf.random.normal(shape=(batch_size,self.input_shape))

class RealImageInput(GanInput):
    def __init__(self,data_config: DataConfig):
        super().__init__(data_config.image_shape,name="real_image_input")
        self.data_helper = DataHelper(data_config)
    
    def load(self):
        print("Preparing Dataset".upper())
        self.images = self.data_helper.load_data()
        self.dataset = tf.data.Dataset.from_tensor_slices(self.images)
        self.dataset_size = len(self.images)
        print("DATASET LOADED")
    
    def save(self,epoch,images,preview_rows,preview_cols,preview_margin):
        self.data_helper.save_images(epoch,images,preview_rows,preview_cols,preview_margin)
    
    def get_batch(self,batch_size):
        d = self.dataset.shuffle(self.dataset_size).batch(batch_size)
        d_iterator = iter(d)
        batch = next(d_iterator)
        while batch.shape[0] != batch_size:
          batch = next(d_iterator)
        return batch
   
