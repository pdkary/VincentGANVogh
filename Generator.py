from keras.layers.convolutional import Conv2DTranspose
from NoiseModel import NoiseModel
from StyleModel import StyleModel
from GanConfig import GenLayerConfig, GeneratorModelConfig, NoiseModelConfig, StyleModelConfig
from keras.layers import UpSampling2D,Conv2D,Input
from keras.models import Functional, Model
import numpy as np
import tensorflow as tf
  
class Generator(GeneratorModelConfig):
    def __init__(self,
                 gen_config: GeneratorModelConfig,
                 noise_config: NoiseModelConfig,
                 style_config: StyleModelConfig):
        GeneratorModelConfig.__init__(self,**gen_config.__dict__)
        
        self.style_model = StyleModel(style_config)
        self.noise_model = NoiseModel(noise_config)
        
        self.gen_constant_input = Input(shape=self.gen_constant_shape, name="gen_constant_input")
        self.gen_constant = tf.random.normal(shape=self.gen_constant_shape)
        
        self.input = [self.gen_constant_input, self.style_model.input, self.noise_model.input]
    
    def get_input(self,batch_size:int):
        return [self.get_constant(batch_size),
                self.style_model.get_noise(batch_size),
                self.noise_model.get_noise(batch_size)]
    
    def get_constant(self,batch_size:int):
        gc_batch = np.full((batch_size,*self.gen_constant_shape),0.0,dtype=np.float32)
        for i in range(batch_size):
            gc_batch[i] = self.gen_constant
        return gc_batch
    
    def build_generator(self):
        out = self.gen_constant_input
        for layer_config in list(self.gen_layers[0])[0]:
            out = self.generator_block(out,layer_config)
        
        gen_model = Model(inputs=self.input,outputs=out,name="Generator")
        gen_model.compile(optimizer=self.gen_optimizer,
                           loss=self.gen_loss_function,
                           metrics=['accuracy'])
        gen_model.summary()
        return gen_model 

    def generator_block(self,input_tensor: Functional,config: GenLayerConfig):
        out = input_tensor
        out = UpSampling2D(interpolation='bilinear')(out) if config.upsampling else out
        for i in range(config.convolutions):
            if config.transpose:
                out = Conv2DTranspose(config.filters,config.kernel_size,config.strides,padding='same', kernel_initializer = 'he_normal')(out)
            else:
                out = Conv2D(config.filters,config.kernel_size,config.strides,padding='same', kernel_initializer = 'he_normal')(out)
            if config.noise:
                out = self.noise_model.add_noise(out,config.filters)
            if config.style:
                out = self.style_model.AdaIN(out,config.filters)
            else:
                out = self.non_style_normalization.get()(out)
            out =  config.activation.get()(out)
        return out