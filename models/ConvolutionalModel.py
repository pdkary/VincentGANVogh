from typing import List

import tensorflow.keras.backend as K
from config.GanConfig import ConvLayerConfig, DiscConvLayerConfig
from layers.CallableConfig import ActivationConfig, NoneCallable, RegularizationConfig
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dropout, Flatten,
                                     GaussianNoise, MaxPooling2D, UpSampling2D)
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

class ConvolutionalModelBuilder():
    def __init__(self,
                 input: KerasTensor,
                 view_channels: int = None,
                 view_activation: ActivationConfig = NoneCallable,
                 std_dims: List[int] = [1,2,3],
                 kernel_regularizer:RegularizationConfig = NoneCallable,
                 kernel_initializer:str = "glorot_uniform"):
        self.out = input
        self.view_channels = view_channels
        self.view_activation = view_activation
        self.std_dims = std_dims
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.tracked_layers = {}
    
    def build(self):
        return self.out
    
    def flatten(self):
        self.out = Flatten()(self.out)
        return self

    def conv_block(self,config:DiscConvLayerConfig):
        if config.upsampling == "stride":
            up_config = ConvLayerConfig(config.filters,1,3,config.activation,transpose=True,strides=(2,2))
            self.out = self.conv_layer(up_config)
        elif config.upsampling == True:
            self.out = UpSampling2D()(self.out)

        self.out = self.conv_layer(config)
        self.out = Dropout(config.dropout_rate)(self.out)
        self.out = config.normalization.get()(self.out)
        self.out = GaussianNoise(config.noise)(self.out) if config.noise > 0.0 else self.out
        self.out = config.activation.get()(self.out)
            
        if config.downsampling == "stride":
            down_config = ConvLayerConfig(config.filters,1,3,config.activation,strides=(2,2))
            self.out = self.conv_layer(down_config)
        elif config.downsampling == True:
            self.out = MaxPooling2D()(self.out)

        return self
    
    def conv_layer(self,config: DiscConvLayerConfig):
        c_args = dict(filters=config.filters,kernel_size=config.kernel_size,
                        strides=config.strides,padding="same",
                        kernel_regularizer = self.kernel_regularizer.get(),
                        kernel_initializer = self.kernel_initializer,
                        use_bias=False)
        for i in range(config.convolutions):
            self.out = Conv2DTranspose(**c_args)(self.out) if config.transpose else Conv2D(**c_args)(self.out)
        if config.track_id != "":
            self.track_current_layer(config.track_id)
        return self.out

    def track_current_layer(self,name:str):
        out_std  = K.std(self.out,self.std_dims,keepdims=True)
        out_mean = K.mean(self.out,self.std_dims,keepdims=True)
        layer_data = {"std_mean":[out_std,out_mean]}
        if self.view_channels is not None:
            viewable_config = DiscConvLayerConfig(self.view_channels,1,1,self.view_activation)
            layer_data["view"] = self.conv_layer(viewable_config)
        self.tracked_layers[name] = layer_data
    