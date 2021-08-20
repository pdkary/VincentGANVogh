from keras.layers import Lambda,Input
from GanConfig import StyleModelConfig
import keras.backend as K
from keras.layers import Dense
import tensorflow as tf

def AdaIN(input_arr):
  input_tensor, gamma, beta = input_arr
  mean = K.mean(input_tensor, axis = [1, 2], keepdims = True)
  std = K.std(input_tensor, axis = [1, 2], keepdims = True) + 1e-7
  y = (input_tensor - mean) / std
  
  pool_shape = [-1, 1, 1, y.shape[-1]]
  scale = K.reshape(gamma, pool_shape)
  bias = K.reshape(beta, pool_shape)
  return y * scale + bias

class StyleModel(StyleModelConfig):
    def __init__(self,style_config):
        super().__init__(**style_config.__dict__)
        self.input = Input(shape=self.style_latent_size, name="style_model_input")
    
    def get_noise(self,batch_size:int):
        return tf.random.normal(shape = (batch_size,self.style_latent_size))
        
    def build(self):
        S = self.input
        for i in range(self.style_layers):
            S = Dense(self.style_latent_size, kernel_initializer = 'he_normal')(S)
            S = self.style_activation.get()(S)
        return S
        