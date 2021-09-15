from abc import ABC, abstractmethod
from typing import Tuple
from tensorflow.keras.regularizers import L2
from layers.AdaptiveAdd import AdaptiveAdd
from tensorflow.keras.layers import Cropping2D, Activation, Input, Cropping2D, Conv2D
import numpy as np
import tensorflow as tf

class NoiseModelBase(ABC):
    def __init__(self,
                 noise_image_size: Tuple[int,int,int],
                 kernel_size: int = 1,
                 max_std_dev: int = 1):
        self.noise_image_size = noise_image_size
        self.kernel_size = kernel_size
        self.max_std_Dev = max_std_dev
        self.input = Input(shape=self.noise_image_size, name="noise_model_input")
        self.model = Activation('linear')(self.input)
    
    def add(self,input_tensor):
        n_size = self.model.shape[1]
        i_size = input_tensor.shape[1]
        noise = Cropping2D((n_size-i_size)//2)(self.model)
        noise = Conv2D(input_tensor.shape[-1],self.kernel_size,padding='same', kernel_regularizer=L2(),kernel_initializer='he_normal')(noise)
        return AdaptiveAdd()([input_tensor,noise])

    @abstractmethod
    def get_batch(self,batch_size:int):
        pass
    

class LatentNoiseModel(NoiseModelBase):
    def __init__(self,noise_config):
        super().__init__(noise_config)
        
    def get_batch(self,batch_size:int):
        noise_batch = np.full((batch_size,*self.noise_image_size),0.0,dtype=np.float32)
        for i in range(batch_size):
            noise_batch[i] = tf.random.normal(shape=self.noise_image_size,stddev=self.max_std_dev)
        return noise_batch
    
class ConstantNoiseModel(NoiseModelBase):
    def __init__(self,noise_config):
        super().__init__(noise_config)
        self.constant = tf.random.normal(shape=self.noise_image_size,stddev=self.max_std_dev)
        
    def get_batch(self,batch_size:int):
        noise_batch = np.full((batch_size,*self.noise_image_size),0.0,dtype=np.float32)
        for i in range(batch_size):
            noise_batch[i] = self.constant
        return noise_batch
