from keras.models import Functional
from GanConfig import NoiseModelConfig
from keras.layers import Input, Activation,Cropping2D,Conv2D,Add
import numpy as np
import tensorflow as tf

class NoiseModel(NoiseModelConfig):
    def __init__(self,noise_config):
        super().__init__(**noise_config.__dict__)
        self.input = Input(shape=self.noise_image_size, name="noise_model_input")
        self.noise_model = Activation('linear')(self.input)
    
    def get_noise(self,batch_size:int):
        noise_batch = np.full((batch_size,*self.noise_image_size),0.0,dtype=np.float32)
        for i in range(batch_size):
            noise_batch[i] = tf.random.normal(shape=self.noise_image_size,stddev=self.gauss_factor)
        return noise_batch

    def add_noise(self, out: Functional, filters: int):
        ## crop noise model to size
        desired_size = out.shape[1]
        noise_size = self.noise_model.shape[1]
        noise_model = Cropping2D((noise_size-desired_size)//2)(self.noise_model)
        ## convolve to match current size
        noise_model = Conv2D(filters,self.noise_kernel_size,padding='same',kernel_initializer='he_normal')(self.noise_model)
        out = Add()([out,noise_model])
