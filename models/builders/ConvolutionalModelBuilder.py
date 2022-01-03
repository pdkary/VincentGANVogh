from typing import List

import tensorflow.keras.backend as K
from config.GanConfig import ConvLayerConfig, DiscConvLayerConfig
from config.CallableConfig import (ActivationConfig, NoneCallable,
                                   RegularizationConfig)
from models.builders.BuilderBase import BuilderBase
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dropout,
                                     GaussianNoise, MaxPooling2D, UpSampling2D)
from tensorflow.python.keras.engine.keras_tensor import KerasTensor


class ConvolutionalModelBuilder(BuilderBase):
    def __init__(self,
                 input_layer: KerasTensor,
                 view_channels: int = None,
                 view_activation: ActivationConfig = NoneCallable,
                 std_dims: List[int] = [1,2,3],
                 kernel_regularizer:RegularizationConfig = NoneCallable,
                 kernel_initializer:str = "glorot_uniform"):
        super().__init__(input_layer)
        self.view_channels = view_channels
        self.view_activation = view_activation
        self.std_dims = std_dims
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer

    """
    convolutional block goes
     - upsampling
     - conv2d
     - dropout
     - normalization
     - noise
     - activation
     - downsampling
    """
    def block(self,config:DiscConvLayerConfig):
        if config.upsampling == "stride":
            up_config = ConvLayerConfig(config.filters,1,3,config.activation,transpose=True,strides=(2,2))
            self.out = self.layer(up_config)
        elif config.upsampling == True:
            self.out = UpSampling2D()(self.out)

        self.out = self.layer(config)
        self.out = Dropout(config.dropout_rate)(self.out)
        self.out = config.normalization.get()(self.out)
        self.out = GaussianNoise(config.noise)(self.out) if config.noise > 0.0 else self.out
        self.out = config.activation.get()(self.out)
            
        if config.downsampling == "stride":
            down_config = ConvLayerConfig(config.filters,1,3,config.activation,strides=(2,2))
            self.out = self.layer(down_config)
        elif config.downsampling == True:
            self.out = MaxPooling2D()(self.out)
        self.layer_count += 1
        return self

    ##shorthand to make a conv2d or conv2d transpose layer
    def layer(self,config: DiscConvLayerConfig):
        c_args = dict(filters=config.filters,kernel_size=config.kernel_size,
                        strides=config.strides,padding="same",
                        kernel_regularizer = self.kernel_regularizer.get(),
                        kernel_initializer = self.kernel_initializer,
                        use_bias=False)
        for i in range(config.convolutions):
            self.out = Conv2DTranspose(**c_args)(self.out) if config.transpose else Conv2D(**c_args)(self.out)
        if config.track_id != "":
            self.track(config.track_id)
        return self.out
    
    ##track the layer in tracked_layers dict
    def track(self,name:str):
        out_std  = K.std(self.out,self.std_dims,keepdims=True)
        out_mean = K.mean(self.out,self.std_dims,keepdims=True)
        layer_data = {"std_mean":[out_std,out_mean]}
        if self.view_channels is not None:
            viewable_config = DiscConvLayerConfig(self.view_channels,1,1,self.view_activation)
            layer_data["view"] = self.layer(viewable_config)
        self.tracked_layers[name] = layer_data
        
    