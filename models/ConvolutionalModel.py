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
        self.model = self.input
        for config in self.conv_layers:
            if config.upsampling:
                self.model = UpSampling2D(interpolation='bilinear')(self.model)
            
            self.conv_block(config)
            
            if config.downsampling:
                self.model = MaxPooling2D()(self.model)
        if flatten:
            self.model = Flatten()(self.model)
        return self.model
    
    def conv_block(self,config:DiscConvLayerConfig):
        for i in range(config.convolutions):
            name = "_".join([config.track_id,str(config.filters),str(i)])
            
            if config.transpose:
                self.model = Conv2DTranspose(**self.get_conv_args(config.filters,config.kernel_size,config.strides))(self.model)
            else:
                self.model = Conv2D(**self.get_conv_args(config.filters,config.kernel_size,config.strides))(self.model)
            
            if config.dropout_rate > 0:
                self.model = Dropout(config.dropout_rate,name="conv_dropout_"+name)(self.model)
            
            if config.noise:
                self.model = GaussianNoise(1.0)(self.model)
            
            if config.style:
                adain = AdaptiveInstanceNormalization(config.filters,name)(self.model)
                self.model = adain
                self.tracked_layers[name] = [adain]
            else:                    
                self.model = config.normalization.get()(self.model)
                act = config.activation.get()(self.model)
                self.model = act 
                self.tracked_layers[name] = [act]