from typing import List

from config.GanConfig import ConvLayerConfig, DiscConvLayerConfig, SimpleActivations
from config.CallableConfig import NoneCallable, RegularizationConfig
from models.builders.BuilderBase import BuilderBase
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dropout, Concatenate, 
                                     GaussianNoise, MaxPooling2D, UpSampling2D)
from tensorflow.python.keras.engine.keras_tensor import KerasTensor


class ConvolutionalModelBuilder(BuilderBase):
    def __init__(self,
                 input_layer: KerasTensor,
                 std_dims: List[int] = [1,2,3],
                 kernel_regularizer:RegularizationConfig = NoneCallable,
                 kernel_initializer:str = "glorot_uniform"):
        super().__init__(input_layer)
        self.std_dims = std_dims
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer

    """
    convolutional block goes brrr
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

        for i in range(config.convolutions):
            self.out = self.layer(config)
            self.out = Dropout(config.dropout_rate)(self.out) if config.dropout_rate > 0.0 else self.out
            self.out = config.normalization.get()(self.out)
            self.out = GaussianNoise(config.noise)(self.out) if config.noise > 0.0 else self.out
            self.out = config.activation.get()(self.out)
            
        if config.downsampling == "stride":
            down_config = ConvLayerConfig(config.filters,1,3,config.activation,strides=(2,2))
            self.out = self.layer(down_config)
        elif config.downsampling == True:
            self.out = MaxPooling2D()(self.out)
            
        ## concatenation for UNET integration
        if config.concat_with != "":
            key = config.concat_with
            if key in self.awaiting_concatenation:
                self.out = Concatenate(axis=-1)([self.out,self.awaiting_concatenation[key]])
                del self.awaiting_concatenation[key]
            else:
                self.awaiting_concatenation[key] = self.out

        if config.view_channels is not None:
            config = DiscConvLayerConfig(config.view_channels,1,1,SimpleActivations.sigmoid.value)
            self.view_layers.append(self.layer(config))
        self.layer_count += 1
        return self

    ##shorthand to make a conv2d or conv2d transpose layer
    def layer(self,config: DiscConvLayerConfig):
        c_args = dict(filters=config.filters,kernel_size=config.kernel_size,
                        strides=config.strides,padding="same",
                        kernel_regularizer = self.kernel_regularizer.get(),
                        kernel_initializer = self.kernel_initializer,
                        use_bias=False)
        self.out = Conv2DTranspose(**c_args)(self.out) if config.transpose else Conv2D(**c_args)(self.out)
        return self.out