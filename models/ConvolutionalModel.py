from typing import List

from tensorflow.python.keras.models import Model

from config.GanConfig import ConvLayerConfig
from inputs.GanInput import GanInput
from layers.AdaptiveInstanceNormalization import AdaINConfig
from layers.CallableConfig import NoneCallable, RegularizationConfig
from tensorflow.python.keras.layers.convolutional import (Conv2D,
                                                          Conv2DTranspose,
                                                          UpSampling2D)
from tensorflow.python.keras.layers.core import Dropout, Flatten
from tensorflow.python.keras.layers.noise import GaussianNoise
from tensorflow.python.keras.layers.pooling import MaxPooling2D

from models.StyleModel import LatentStyleModel

class ConvolutionalModel():
    def __init__(self,
                 gan_input: GanInput,
                 conv_layers: List[ConvLayerConfig],
                 style_model: LatentStyleModel = None,
                 kernel_regularizer:RegularizationConfig = NoneCallable,
                 kernel_initializer:str = "glorot_uniform"):
        self.gan_input = gan_input
        self.conv_layers = conv_layers
        self.style_model = style_model
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.tracked_layers = {}
        self.inputs = [gan_input.input,style_model.dense_input] if style_model is not None else [gan_input.input]

    def get_conv(self,config: ConvLayerConfig):
        if config.transpose:
            return Conv2DTranspose(config.filters,config.kernel_size,config.strides,
                                padding='same',kernel_regularizer=self.kernel_regularizer.get(), 
                                kernel_initializer = self.kernel_initializer, use_bias=False)
        else:
            return Conv2D(config.filters,config.kernel_size,
                        padding='same',kernel_regularizer=self.kernel_regularizer.get(), 
                        kernel_initializer = self.kernel_initializer, use_bias=False)
    
    def build(self,flatten=False):
        out = self.gan_input.input

        if self.style_model is not None:
            style_out = self.style_model.build()

        for config in self.conv_layers:
            out = UpSampling2D(interpolation='bilinear')(out) if config.upsampling else out
            for i in range(config.convolutions):
                name = "_".join([config.track_id,str(config.filters),str(i)])
                out = self.get_conv(config)(out)
                out = Dropout(config.dropout_rate,name="conv_dropout_"+name)(out)
                out = GaussianNoise(1.0)(out) if config.noise else out
                
                if config.style and self.style_model is not None:
                    out,beta,gamma = AdaINConfig().get(style_out,config.filters,name)(out)
                    self.tracked_layers[name] = [beta,gamma]
                else:
                    out = config.normalization.get()(out)
                        
                out = config.activation.get()(out)
                if config.track_id != "":
                    self.tracked_layers[name] = [out]
            out = MaxPooling2D()(out) if config.downsampling else out
        
        out = Flatten(name="conv_flatten_" + name)(out) if flatten else out
        return out
    
    def get_training_batch(self,batch_size):
        return self.gan_input.get_training_batch(batch_size)

    def get_validation_batch(self,batch_size):
        return self.gan_input.get_validation_batch(batch_size)