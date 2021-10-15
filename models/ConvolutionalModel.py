from re import A
from typing import List
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from config.GanConfig import ConvLayerConfig, DiscConvLayerConfig
from layers.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization
from layers.CallableConfig import NoneCallable, RegularizationConfig
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dropout, Flatten,
                                     GaussianNoise, MaxPooling2D, UpSampling2D)


class ConvolutionalModel():
    def __init__(self,
                 input: KerasTensor,
                 conv_layers: List[ConvLayerConfig],
                 kernel_regularizer:RegularizationConfig = NoneCallable,
                 kernel_initializer:str = "glorot_uniform"):
        self.input = input
        self.conv_layers = conv_layers
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.tracked_layers = {}

    def get_conv_args(self,filters,kernel_size,strides):
        return dict(filters=filters,kernel_size=kernel_size,strides=strides,
                     padding='same',kernel_regularizer=self.kernel_regularizer.get(),
                     kernel_initializer=self.kernel_initializer, use_bias=False)

    def build(self,flatten=False):
        #configure input
        out = self.input
        for config in self.conv_layers:
            out = self.conv_block(out,config)
        
        if flatten:
            out = Flatten()(out)
        return out
    
    def conv_block(self,input_tensor: KerasTensor,config:DiscConvLayerConfig):
        out = input_tensor
        if config.upsampling:
            out = UpSampling2D(interpolation='bilinear')(out)
        for i in range(config.convolutions):
            name = "_".join([config.track_id,str(config.filters),str(i)])
            if config.transpose:
                out = Conv2DTranspose(filters=config.filters,kernel_size=config.kernel_size,
                                      strides=config.strides,padding='same',
                                      kernel_regularizer=self.kernel_regularizer.get(),
                                      kernel_initializer=self.kernel_initializer, use_bias=False)(out)
            else:
                out = Conv2D(filters=config.filters,kernel_size=config.kernel_size,
                             strides=config.strides,padding='same',
                             kernel_regularizer=self.kernel_regularizer.get(),
                             kernel_initializer=self.kernel_initializer, use_bias=False)(out)
            
            if config.dropout_rate > 0:
                out = Dropout(config.dropout_rate,name="conv_dropout_"+name)(out)
            
            if config.noise:
                out = GaussianNoise(1.0)(out)
            
            if config.style:
                out,B,G = AdaptiveInstanceNormalization(config.filters,name)(out)
                self.tracked_layers[name] = [B,G]
            else:                    
                out = config.normalization.get()(out)
                out = config.activation.get()(out)
                self.tracked_layers[name] = [out]
        if config.downsampling:
            out = MaxPooling2D()(out)
        return out
