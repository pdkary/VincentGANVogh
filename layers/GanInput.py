from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
from config.GanConfig import ActivationConfig
from config.TrainingConfig import DataConfig
from helpers.DataHelper import DataHelper
from tensorflow.keras.backend import prod
from tensorflow.keras.layers import Activation, Dense, Input, Reshape
from tensorflow.keras.regularizers import L2


class GanInput(ABC):
    def __init__(self,input_shape: Tuple[int,int,int],name="gan_input"):
        self.input_shape = input_shape
        self.input = Input(shape=input_shape,dtype=tf.float32,name=name)
        self.model = Activation('linear')(self.input)
    
    @abstractmethod    
    def get_training_batch(self,batch_size):
        pass
    
    @abstractmethod    
    def get_validation_batch(self,batch_size):
        pass

class EncoderInput(GanInput):
    def __init__(self,input_shape: int):
        super().__init__((input_shape,),name="encoded_input")
    
class GenConstantInput(GanInput):
    def __init__(self, input_shape: Tuple):
        super().__init__(input_shape,name="gen_constant_input")
        self.constant = tf.constant(tf.random.normal(shape=input_shape,dtype=tf.float32))
        self.model = Activation('linear')(self.input)
    
    def get_training_batch(self, batch_size):
        gc_batch = np.full((batch_size,*self.input_shape),0.0,dtype=np.float32)
        for i in range(batch_size):
            gc_batch[i] = self.constant
        return gc_batch
    
    def get_validation_batch(self, batch_size):
        return self.get_training_batch(batch_size)
        
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
    
    def get_training_batch(self, batch_size):
        return tf.random.normal(shape=(batch_size,self.input_shape))
    
    def get_validation_batch(self, batch_size):
        return self.get_training_batch(batch_size)

class RealImageInput(GanInput,DataConfig):
    def __init__(self,data_config: DataConfig):
        GanInput.__init__(self,data_config.image_shape,name="real_image_input")
        DataConfig.__init__(self,**data_config.__dict__)
        self.data_helper = DataHelper(data_config)
        print("Preparing Dataset".upper())
        self.images = self.data_helper.load_data()
        
        self.num_training_imgs = len(self.images)//2
        self.training_images = self.images[:self.num_training_imgs]
        self.validation_images = self.images[self.num_training_imgs:]
        self.training_dataset = tf.data.Dataset.from_tensor_slices(self.training_images)
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(self.validation_images)
        print("DATASET LOADED")
    
    def save(self,epoch,images,preview_rows,preview_cols,preview_margin):
        self.data_helper.save_images(epoch,images,preview_rows,preview_cols,preview_margin)
    
    def get_training_batch(self,batch_size):
        return self.get_batch(batch_size,self.training_dataset)
    
    def get_validation_batch(self,batch_size):
        return self.get_batch(batch_size,self.validation_dataset)
    
    def get_batch(self,batch_size:int,dataset:Dataset):
        d = dataset.shuffle(self.num_training_imgs//2).batch(batch_size)
        d_iterator = iter(d)
        batch = next(d_iterator)
        while batch.shape[0] != batch_size:
          batch = next(d_iterator)
        return batch
   
