from models.GanInput import RealImageInput
import numpy as np
from config.GeneratorConfig import GeneratorModelConfig, GenLayerConfig
from keras.layers import Conv2D, Dense, UpSampling2D
from keras.layers.convolutional import Conv2DTranspose
from keras.models import Model
from layers.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization
from models.Generator.NoiseModel import NoiseModel
from models.Generator.StyleModel import StyleModel

class Generator(GeneratorModelConfig):
    def __init__(self,gen_config: GeneratorModelConfig,batch_size: int, preview_size: int):
        GeneratorModelConfig.__init__(self,**gen_config.__dict__)
        self.batch_size = batch_size
        self.preview_size = preview_size
        self.using_style = np.any([l.style for l in list(self.gen_layers[0])[0]])
        self.using_noise = np.any([l.noise for l in list(self.gen_layers[0])[0]])
        self.using_image_input = isinstance(self.input_model,RealImageInput)
        
        self.input = [self.input_model.input]
        if self.using_image_input:
            self.input_model.load()
             
        if self.using_style:
            self.style_model = StyleModel(self.style_model_config)
            self.input.append(self.style_model.input)
            
        if self.using_noise:
            self.noise_model = NoiseModel(self.noise_model_config)
            self.input.append(self.noise_model.input)
    
    def get_input(self,training=True):
        batch_size = self.batch_size if training else self.preview_size
        if self.using_image_input:
            inp = [self.input_model.get_batch(training)]
        else:
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
