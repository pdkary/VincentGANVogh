from models.GeneratorInput import GenConstantInput
from layers.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization
from keras.layers.convolutional import Conv2DTranspose
from models.NoiseModel import NoiseModel
from models.StyleModel import StyleModel
from models.GanConfig import GenLayerConfig, GeneratorModelConfig, NoiseModelConfig, StyleModelConfig
from keras.layers import UpSampling2D, Conv2D, Dense
from keras.models import Model
import numpy as np
  
class Generator(GeneratorModelConfig):
    def __init__(self,gen_config: GeneratorModelConfig):
        GeneratorModelConfig.__init__(self,**gen_config.__dict__)
        
        self.using_style = np.any([l.style for l in list(self.gen_layers[0])[0]])
        self.using_noise = np.any([l.noise for l in list(self.gen_layers[0])[0]])
        
        self.input = [self.input_model.input] 
        if self.using_style:
            self.style_model = StyleModel(self.style_model_config)
            self.input.append(self.style_model.input)
            
        if self.using_noise:
            self.noise_model = NoiseModel(self.noise_model_config)
            self.input.append(self.noise_model.input)
    
    def get_input(self,batch_size:int):
        inp = [self.input_model.get_batch(batch_size)]
        if self.using_style:
            inp.append(self.style_model.get_batch(batch_size))
        if self.using_noise:
            inp.append(self.noise_model.get_batch(batch_size))
        return inp
    
    def build_generator(self):
        out = self.input_model.model
        for layer_config in list(self.gen_layers[0])[0]:
            out = self.generator_block(out,layer_config)
        
        gen_model = Model(inputs=self.input,outputs=out,name="Generator")
        gen_model.compile(optimizer=self.gen_optimizer,
                           loss="binary_crossentropy",
                           metrics=['accuracy'])
        gen_model.summary()
        return gen_model 

    def generator_block(self,input_tensor,config: GenLayerConfig):
        out = input_tensor
        out = UpSampling2D(interpolation='bilinear')(out) if config.upsampling else out
        for i in range(config.convolutions):
            if config.transpose:
                out = Conv2DTranspose(config.filters,config.kernel_size,config.strides,padding='same', kernel_initializer = 'he_normal')(out)
            else:
                out = Conv2D(config.filters,config.kernel_size,config.strides,padding='same', kernel_initializer = 'he_normal')(out)
            
            if config.noise:
                out = self.noise_model.add(out)
            
            if config.style:
                gamma = Dense(config.filters,bias_initializer='ones')(self.style_model.model)
                beta = Dense(config.filters,bias_initializer='zeros')(self.style_model.model)
                out = AdaptiveInstanceNormalization()([out,gamma,beta])
            else:
                out = self.non_style_normalization.get()(out)
            out =  config.activation.get()(out)
        return out