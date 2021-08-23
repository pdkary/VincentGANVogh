from config.GeneratorConfig import NoiseModelConfig
from layers.AdaptiveAdd import AdaptiveAdd
from keras.layers.convolutional import Cropping2D
from keras.layers.core import Activation
from keras.layers import Input,Cropping2D,Conv2D
import numpy as np
import tensorflow as tf

class NoiseModel(NoiseModelConfig):
    def __init__(self,noise_config):
        super().__init__(**noise_config.__dict__)
        self.input = Input(shape=self.noise_image_size, name="noise_model_input")
        self.model = Activation('linear')(self.input)
        
    def get_batch(self,batch_size:int):
        noise_batch = np.full((batch_size,*self.noise_image_size),0.0,dtype=np.float32)
        for i in range(batch_size):
            noise_batch[i] = tf.random.normal(shape=self.noise_image_size,stddev=self.gauss_factor)
        return noise_batch

    def add(self,input_tensor):
        n_size = self.model.shape[1]
        i_size = input_tensor.shape[1]
        noise = Cropping2D((n_size-i_size)//2)(self.model)
        noise = Conv2D(input_tensor.shape[-1],self.kernel_size,padding='same',kernel_initializer='he_normal')(noise)
        return AdaptiveAdd()([input_tensor,noise])