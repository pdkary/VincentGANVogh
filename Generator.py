from keras.layers.convolutional import Cropping2D
from keras.layers.core import Activation
from GanConfig import GeneratorModelConfig, NoiseModelConfig, StyleModelConfig
from keras.layers import UpSampling2D,Conv2D,Dense,Add,Lambda,LeakyReLU,Input
from keras.models import Model
import keras.backend as K
import numpy as np
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
  
class Generator(GeneratorModelConfig,NoiseModelConfig,StyleModelConfig):
    def __init__(self,gen_config,noise_config,style_config):
        GeneratorModelConfig.__init__(self,**gen_config.__dict__)
        NoiseModelConfig.__init__(self,**noise_config.__dict__)
        StyleModelConfig.__init__(self,**style_config.__dict__)

        self.gen_constant_input = Input(shape=self.gen_constant_shape, name="gen_constant_input")
        self.style_model_input = Input(shape=self.style_latent_size, name="style_model_input")
        self.noise_model_input = Input(shape=self.noise_image_size, name="noise_model_input")
        
        self.gen_constant = tf.random.normal(shape=self.gen_constant_shape)
        
        self.input = [self.gen_constant_input, self.style_model_input, self.noise_model_input]
    
    def get_input(self,batch_size):
        return [self.get_constant(batch_size),
                self.get_style_noise(batch_size),
                self.get_noise(batch_size)]
    
    def get_constant(self,batch_size):
        gc_batch = np.full((batch_size,*self.gen_constant_shape),0.0,dtype=np.float32)
        for i in range(batch_size):
            gc_batch[i] = self.gen_constant
        return gc_batch

    def get_style_noise(self,batch_size):
        return tf.random.normal(shape = (batch_size,self.style_latent_size))
        
    def get_noise(self,batch_size):
        noise_batch = np.full((batch_size,*self.noise_image_size),0.0,dtype=np.float32)
        for i in range(batch_size):
            noise_batch[i] = tf.random.normal(shape=self.noise_image_size,stddev=self.gauss_factor)
        return noise_batch
    
    def build(self):
        S = self.build_style_model()
        N = self.build_noise_model()
        return self.build_generator(S,N)
    
    def build_noise_model(self):
        return Activation('linear')(self.noise_model_input)
    
    def build_style_model(self):
        out = self.style_model_input
        for i in range(self.style_layers):
            out = Dense(self.style_latent_size, kernel_initializer = 'he_normal')(out)
            out = self.style_activation(out)
        return out 
    
    def build_generator(self,style_model,noise_model):
        out = self.gen_constant_input
        for shape,upsampling,noise,style in zip(self.gen_layer_shapes,self.gen_layer_upsampling,self.gen_layer_noise,self.gen_layer_using_style):
            out = self.generator_block(out,style_model,noise_model,*shape,upsampling=upsampling,style=style,noise=noise)
        out = Conv2D(self.img_shape[-1], self.gen_kernel_size, padding='same',activation='sigmoid')(out)
        
        gen_model = Model(inputs=self.input,outputs=out,name="Generator")
        gen_model.compile(optimizer=self.gen_optimizer,
                           loss=self.gen_loss_function,
                           metrics=['accuracy'])
        gen_model.summary()
        return gen_model 

    def generator_block(self,input_tensor,style_model,noise_model,filters,convolutions,upsampling=True,style=True,noise=True):
        out = input_tensor
        out = UpSampling2D(interpolation='bilinear')(out) if upsampling else out
        for i in range(convolutions):
            out = Conv2D(filters,self.gen_kernel_size,padding='same', kernel_initializer = 'he_normal')(out)
            if noise:
                ## crop noise model to size
                desired_size = out.shape[1]
                noise_size = noise_model.shape[1]
                noise_model = Cropping2D((noise_size-desired_size)//2)(noise_model)
                noise_model = Conv2D(filters,self.noise_kernel_size,padding='same',kernel_initializer='he_normal')(noise_model)
                out = Add()([out,noise_model])
            if style:
                gamma = Dense(filters,bias_initializer='ones')(style_model)
                beta = Dense(filters,bias_initializer='zeros')(style_model)
                out = Lambda(AdaIN)([out,gamma,beta]) 
            else:
                out = self.non_style_normalization_layer(out)
            out =  self.convolution_activation(out)
        return out