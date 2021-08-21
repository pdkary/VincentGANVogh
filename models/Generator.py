from models.GeneratorInput import GenConstantInput
from layers.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization
from keras.layers.convolutional import Conv2DTranspose
from models.NoiseModel import NoiseModel
from models.StyleModel import StyleModel
from models.GanConfig import GenLayerConfig, GeneratorModelConfig, NoiseModelConfig, StyleModelConfig
from keras.layers import UpSampling2D, Conv2D, Dense
from keras.models import Model
  
class Generator(GeneratorModelConfig):
    def __init__(self,
                 gen_config: GeneratorModelConfig,
                 noise_config: NoiseModelConfig,
                 style_config: StyleModelConfig):
        GeneratorModelConfig.__init__(self,**gen_config.__dict__)
        
        self.style_model = StyleModel(style_config)
        self.noise_model = NoiseModel(noise_config)
        
        self.input = [self.input_model.input, self.style_model.input, self.noise_model.input]
    
    def get_input(self,batch_size:int):
        return [self.input_model.get_batch(batch_size),
                self.style_model.get_batch(batch_size),
                self.noise_model.get_batch(batch_size)]
    
    def build_generator(self):
        out = self.input_model.model
        for layer_config in list(self.gen_layers[0])[0]:
            out = self.generator_block(out,layer_config)
        
        gen_model = Model(inputs=self.input,outputs=out,name="Generator")
        gen_model.compile(optimizer=self.gen_optimizer,
                           loss=self.gen_loss_function,
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