from keras.layers.core import Activation
from GanConfig import NoiseModelConfig
from keras.layers import Input
import numpy as np
import tensorflow as tf

class NoiseModel(NoiseModelConfig):
    def __init__(self,noise_config):
        super().__init__(**noise_config.__dict__)
        self.input = Input(shape=self.noise_image_size, name="noise_model_input")
    
    def get_noise(self,batch_size:int):
        noise_batch = np.full((batch_size,*self.noise_image_size),0.0,dtype=np.float32)
        for i in range(batch_size):
            noise_batch[i] = tf.random.normal(shape=self.noise_image_size,stddev=self.gauss_factor)
        return noise_batch

    def build(self):
        return Activation('linear')(self.input)