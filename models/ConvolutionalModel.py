from re import A
from typing import List
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from config.GanConfig import ConvLayerConfig, DiscConvLayerConfig
from layers.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization
from layers.CallableConfig import NoneCallable, RegularizationConfig
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dropout, Flatten,
                                     GaussianNoise, MaxPooling2D, UpSampling2D)

import tensorflow.keras.backend as K

class ConvolutionalModel():
    def __init__(self,
                 input: KerasTensor,
                 conv_layers: List[ConvLayerConfig],
                 style_input: KerasTensor = None,
                 view_channels: int = None,
                 kernel_regularizer:RegularizationConfig = NoneCallable,
                 kernel_initializer:str = "glorot_uniform"):
        self.input = input
        self.conv_layers = conv_layers
        self.style_input = style_input
        self.view_channels = view_channels
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.tracked_layers = {}
        self.viewing_layers = []
    
    def conv_layer(self,config: DiscConvLayerConfig):
        c_args = dict(filters=config.filters,kernel_size=config.kernel_size,
                        strides=config.strides,padding="same",
                        kernel_regularizer = self.kernel_regularizer.get(),
                        kernel_initializer = self.kernel_initializer)
        return Conv2DTranspose(**c_args) if config.transpose else Conv2D(**c_args)

    def build(self,flatten=False):
        #configure input
        out = self.input
        print("BUILDING CONV MODEL")
        for config in self.conv_layers:
            print("BLOCK SHAPE: ",out.shape)
            out = self.conv_block(out,config)
        print("BLOCK SHAPE: ",out.shape)
        if flatten:
            out = Flatten()(out)
        return out
    
    def conv_block(self,input_tensor: KerasTensor,config:DiscConvLayerConfig):
        out = input_tensor
        if config.upsampling:
            out = UpSampling2D(interpolation='bilinear')(out)
        
        for i in range(config.convolutions):
            name = "_".join([config.track_id,str(config.filters),str(i)])
            out = self.conv_layer(config)(out)
            #viewable output layers
            if i == config.convolutions -1 and self.view_channels is not None:
                viewable_config = DiscConvLayerConfig(self.view_channels,1,1,self.conv_layers[-1].activation)
                viewable_out = self.conv_layer(viewable_config)(out)
                self.viewing_layers.append(viewable_out)
            
            if config.dropout_rate > 0:
                out = Dropout(config.dropout_rate,name="conv_dropout_"+name)(out)
            
            if config.noise > 0.0:
                out = GaussianNoise(config.noise)(out)
            ##style
            if config.style and self.style_input is not None:
                out,B,G = AdaptiveInstanceNormalization(config.filters,name)([out,self.style_input])
                out = config.activation.get()(out)
                ##tracking
                if i == config.convolutions - 1 and config.track_id != "":
                    self.tracked_layers[name] = [B,G]
            ##nonstyle
            else:                    
                out = config.normalization.get()(out)
                out = config.activation.get()(out)
                
                if i == config.convolutions - 1 and config.track_id != "":
                    out_std  = K.std(out,[1,2],keepdims=True)
                    out_mean = K.mean(out,[1,2],keepdims=True)
                    self.tracked_layers[name] = [out_std,out_mean]
        if config.downsampling:
            out = MaxPooling2D()(out)
        return out
