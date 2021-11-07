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
                 view_channels: int = None,
                 std_dims: List[int] = [1,2,3],
                 kernel_regularizer:RegularizationConfig = NoneCallable,
                 kernel_initializer:str = "glorot_uniform"):
        self.input = input
        self.conv_layers = conv_layers
        self.view_channels = view_channels
        self.std_dims = std_dims
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.tracked_layers = {}
    
    def conv_layer(self,config: DiscConvLayerConfig,input_tensor: KerasTensor):
        c_args = dict(filters=config.filters,kernel_size=config.kernel_size,
                        strides=config.strides,padding="same",
                        kernel_regularizer = self.kernel_regularizer.get(),
                        kernel_initializer = self.kernel_initializer,
                        use_bias=False)
        out = input_tensor
        for i in range(config.convolutions):
            name = "_".join([config.track_id,str(config.filters),str(i)])
            out = Conv2DTranspose(**c_args)(out) if config.transpose else Conv2D(**c_args)(out)
            if i == config.convolutions - 1 and config.track_id != "":
                self.track_layer(out,name)
        return out

    def build(self,flatten=False):
        out = self.input
        for config in self.conv_layers:
            out = self.conv_block(out,config)
        if flatten:
            out = Flatten()(out)
        return out
    
    def conv_block(self,input_tensor: KerasTensor,config:DiscConvLayerConfig):
        out = input_tensor
        if config.upsampling == "stride":
            downsample_config = ConvLayerConfig(config.filters,1,3,config.activation,transpose=True,strides=(2,2))
            out = self.conv_layer(downsample_config,out)
        elif config.upsampling == True:
            out = UpSampling2D()(out)

        out = self.conv_layer(config,out)
        out = Dropout(config.dropout_rate)(out)
        out = config.normalization.get()(out)
        
        if config.noise > 0.0:
            out = GaussianNoise(config.noise)(out)

        out = config.activation.get()(out)
            
        if config.downsampling == "stride" or config.downsampling == True:
            out = MaxPooling2D()(out)

        return out
    
    def track_layer(self,tensor: KerasTensor,name:str):
        out_std  = K.std(tensor,self.std_dims,keepdims=True)
        out_mean = K.mean(tensor,self.std_dims,keepdims=True)
        layer_data = {"std_mean":[out_std,out_mean]}
        if self.view_channels is not None:
            viewable_config = DiscConvLayerConfig(self.view_channels,1,1,self.conv_layers[-1].activation)
            layer_data["view"] = self.conv_layer(viewable_config,tensor)
        self.tracked_layers[name] = layer_data
    