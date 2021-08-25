from tensorflow._api.v2 import data
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
    def __init__(self,input_shape: Tuple[int,int,int],name="gan_input"):
        self.input_shape = input_shape
        self.input = Input(shape=input_shape,dtype=tf.float32,name=name)
        self.model = Activation('linear')(self.input)
        
    def get_batch(self,batch_size,training=True):
        pass

class GenConstantInput(GanInput):
    def __init__(self, input_shape: Tuple):
        super().__init__(input_shape,name="gen_constant_input")
        self.constant = tf.constant(tf.random.normal(shape=input_shape,dtype=tf.float32))
        self.model = Activation('linear')(self.input)
    
    def get_batch(self, batch_size,training=True):
        gc_batch = np.full((batch_size,*self.input_shape),0.0,dtype=np.float32)
        for i in range(batch_size):
            gc_batch[i] = self.constant
        return gc_batch
        
class GenLatentSpaceInput(GanInput):
    def __init__(self, input_shape: int,output_shape:Tuple,layer_size,layers):
        super().__init__(input_shape,name="gen_latent_space_input")
        self.model = self.input
        for i in range(layers):
            self.model = Dense(layer_size,kernel_initializer="he_normal")(self.model)
        self.model = Dense(prod(output_shape),kernel_initializer="he_normal")(self.model)
        self.model = Reshape(output_shape)(self.model)
    
    def get_batch(self, batch_size,training=True):
        return tf.random.normal(shape=(batch_size,self.input_shape))

class RealImageInput(GanInput):
    def __init__(self,data_config: DataConfig):
        super().__init__(data_config.image_shape,name="real_image_input")
        self.data_helper = DataHelper(data_config)
        self.preview_size = data_config.preview_cols*data_config.preview_rows
    
    def load(self):
        print("Preparing Dataset".upper())
        self.images = self.data_helper.load_data()
        dataset = tf.data.Dataset.from_tensor_slices(self.images)
        self.train_dataset = dataset.batch(self.data_helper.batch_size)
        self.preview_dataset = dataset.batch(self.preview_size)
        self.dataset_size = len(self.images)
        print("DATASET LOADED")
    
    def save(self,epoch,images):
        self.data_helper.save_images(epoch,images)
    
    def get_batch(self,training=True):
        batch_size = self.data_helper.batch_size if training else self.preview_size
        d = self.train_dataset if training else self.preview_dataset
        d = d.shuffle(self.dataset_size)
        d_iterator = iter(d)
        batch = next(d_iterator)
        while batch.shape[0] != batch_size:
          batch = next(d_iterator)
        return batch
   
