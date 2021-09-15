from tensorflow.keras.regularizers import L2
import numpy as np
from config.GeneratorConfig import GeneratorModelConfig, GenLayerConfig
from tensorflow.keras.layers import Conv2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Model
from layers.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization
from models.NoiseModel import NoiseModel
from models.StyleModel import StyleModel

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
    
    def get_input(self,batch_size):
        inp = [self.input_model.get_batch(batch_size)]
        if self.using_style:
            inp.append(self.style_model.get_batch(batch_size))
        if self.using_noise:
            inp.append(self.noise_model.get_batch(batch_size))
        return inp
    
    def build(self,print_summary=True):
        out = self.input_model.model
        for layer_config in list(self.gen_layers[0])[0]:
            out = self.generator_block(out,layer_config)
        self.functional_model = out
        gen_model = Model(inputs=self.input,outputs=out,name="generator")
        gen_model.compile(optimizer=self.gen_optimizer,
                           loss=self.loss_function,
                           metrics=['accuracy'])
        if print_summary:
            gen_model.summary()
        return gen_model


    def generator_block(self,input_tensor,config: GenLayerConfig):
        out = input_tensor
        out = UpSampling2D(interpolation='bilinear')(out) if config.upsampling else out
        for i in range(config.convolutions):
            if config.transpose:
                out = Conv2DTranspose(config.filters,config.kernel_size,config.strides,padding='same',kernel_regularizer=L2(), kernel_initializer = 'he_normal')(out)
            else:
                out = Conv2D(config.filters,config.kernel_size,padding='same',kernel_regularizer=L2(), kernel_initializer = 'he_normal')(out)
            
            if config.noise:
                out = self.noise_model.add(out)
            
            if config.style:
                out = AdaptiveInstanceNormalization()(out,self.style_model.model)
            else:
                out = self.normalization.get()(out)
            out =  config.activation.get()(out)
        return out
