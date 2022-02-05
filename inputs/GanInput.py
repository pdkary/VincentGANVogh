from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow.keras.layers import Input
from config.TrainingConfig import DataConfig
from helpers.DataHelper import DataHelper

class GanInput(ABC):
    def __init__(self,input_shape: Tuple[int,int,int],name="input",train_test_ratio=0.5):
        self.input_shape = input_shape
        self.input_layer: Input = Input(shape=input_shape,name=name)
        self.model_name = name
        self.data_path = None
        self.train_test_ratio = train_test_ratio
    
    @abstractmethod    
    def get_training_batch(self,batch_size):
        pass

    def get_validation_batch(self,batch_size):
        return self.get_training_batch(batch_size)
    
    def save(self,name:str):
        pass

class ConstantInput(GanInput):
    def __init__(self, input_shape: Tuple[int,int,int]):
        super().__init__(input_shape,name="constant_input")
        self.constant = np.random.normal(size=input_shape)
    
    def get_training_batch(self, batch_size):
        gc_batch = np.full((batch_size,*self.input_shape),0.0,dtype=np.float32)
        for i in range(batch_size):
            gc_batch[i] = self.constant
        return gc_batch
    
class LatentSpaceInput(GanInput):
    def __init__(self, input_shape:Tuple,name="latent_space_input"):
        super().__init__(input_shape,name=name)
    
    def get_training_batch(self, batch_size):
        return tf.random.normal(shape=(batch_size,*self.input_shape))
    
class RealImageInput(GanInput,DataConfig):
    def __init__(self,data_config: DataConfig,train_test_ratio=0.5):
        GanInput.__init__(self,data_config.image_shape,name="real_image_input",train_test_ratio=train_test_ratio)
        DataConfig.__init__(self,**data_config.__dict__)
        self.data_helper = DataHelper(data_config)
        self.images = self.data_helper.load_data()
        
        self.num_training_imgs = len(self.images)*self.train_test_ratio
        self.training_images = np.asarray(self.images[:self.num_training_imgs],np.float32)
        self.validation_images = np.asarray(self.images[self.num_training_imgs:],np.float32)
        self.training_dataset = tf.data.Dataset.from_tensor_slices(self.training_images)
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(self.validation_images)
        print("DATASET LOADED")

    def get_training_batch(self,batch_size):
        return self.get_batch(batch_size,self.training_dataset)
    
    def get_validation_batch(self,batch_size):
        return self.get_batch(batch_size,self.validation_dataset)
    
    def get_batch(self,batch_size:int,dataset:Dataset):
        d = dataset.shuffle(self.num_training_imgs//2).batch(batch_size)
        d0 =  [x for x in d.take(1)]
        return d0[0]