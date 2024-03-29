from typing import List, Union

from config.GanConfig import ConvLayerConfig
from inputs.GanInput import GanInput
from layers.AdaptiveAdd import AdaptiveAdd
from layers.AdaptiveInstanceNormalization import AdaINConfig
from layers.CallableConfig import NoneCallable, RegularizationConfig
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dropout, Flatten,
                                     GaussianNoise, MaxPooling2D, UpSampling2D)

from models.DenseModel import LatentSpaceModel


class ConvolutionalModel():
    def __init__(self,
                 gan_input: Union[GanInput,LatentSpaceModel],
                 conv_layers: List[ConvLayerConfig],
                 style_model: LatentSpaceModel = None,
                 kernel_regularizer:RegularizationConfig = NoneCallable,
                 kernel_initializer:str = "glorot_uniform"):
        self.gan_input = gan_input
        self.conv_layers = conv_layers
        self.style_model = style_model
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.tracked_layers = {}
        gen_input = gan_input.inputs if isinstance(self.gan_input,LatentSpaceModel) else gan_input.input
        self.inputs = [gen_input,style_model.inputs] if style_model is not None else [gen_input]

    def get_conv_args(self,filters,kernel_size,strides):
        return dict(filters=filters,kernel_size=kernel_size,strides=strides,
                     padding='same',kernel_regularizer=self.kernel_regularizer.get(),
                     kernel_initializer=self.kernel_initializer, use_bias=False)

    def build(self,flatten=False):
        out = self.gan_input.input if isinstance(self.gan_input,GanInput) else self.gan_input.build()

        if self.style_model is not None:
            style_out = self.style_model.build()

        for config in self.conv_layers:
            out = UpSampling2D(interpolation='bilinear')(out) if config.upsampling else out
            for i in range(config.convolutions):
                name = "_".join([config.track_id,str(config.filters),str(i)])
                
                if config.transpose:
                    out = Conv2DTranspose(**self.get_conv_args(config.filters,config.kernel_size,config.strides))(out)
                else:
                    out = Conv2D(**self.get_conv_args(config.filters,config.kernel_size,config.strides))(out)
                out = Dropout(config.dropout_rate,name="conv_dropout_"+name)(out) if config.dropout_rate > 0 else out
                
                if config.noise:
                    noise = GaussianNoise(1.0)(out)
                    out = AdaptiveAdd()([out,noise])
                
                if config.style and self.style_model is not None:
                    out,beta,gamma = AdaINConfig().get(style_out,config.filters,name)(out)
                    out = config.activation.get()(out)  
                    self.tracked_layers[name] = [beta,gamma]
                else:                    
                    out = config.normalization.get()(out)
                    out = config.activation.get()(out)  
                    self.tracked_layers[name] = [out]
            out = MaxPooling2D()(out) if config.downsampling else out
        
        out = Flatten(name="conv_flatten_" + name)(out) if flatten else out
        return out
    
    def get_training_batch(self,batch_size):
        return self.gan_input.get_training_batch(batch_size)

    def get_validation_batch(self,batch_size):
        return self.gan_input.get_validation_batch(batch_size)
