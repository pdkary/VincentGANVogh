from typing import Tuple, List

from tensorflow.python.keras.metrics import Metric

from config.GanConfig import GenLayerConfig
from layers.CallableConfig import NormalizationConfig, RegularizationConfig, NoneCallable
from layers.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization
from layers.GanInput import GanInput

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense,UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.keras.losses import Loss

from models.NoiseModel import NoiseModelBase
from models.StyleModel import StyleModelBase


class Generator():
    def __init__(self,
                 img_shape: Tuple[int,int,int],
                 input_model: GanInput,
                 gen_layers: List[GenLayerConfig],
                 gen_optimizer: Optimizer,
                 loss_function: Loss,
                 metrics: List[Metric],
                 style_model: StyleModelBase = None,
                 noise_model: NoiseModelBase = None,
                 normalization: NormalizationConfig = NoneCallable,
                 kernel_regularizer:RegularizationConfig = NoneCallable,
                 kernel_initializer:str = "glorot_uniform"):
        self.img_shape = img_shape
        self.input_model = input_model
        self.gen_layers = gen_layers
        self.gen_optimizer = gen_optimizer
        self.loss_function = loss_function
        self.metrics = [m() for m in metrics]
        self.metric_labels = ["G_" + str(m.name) for m in self.metrics]
        self.style_model = style_model
        self.noise_model = noise_model
        self.normalization = normalization
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.layer_sizes = [self.input_model.input_shape]
        
        self.input = [self.input_model.input]
        if self.style_model is not None:
            self.input.append(self.style_model.input)
            
        if self.noise_model is not None:
            self.input.append(self.noise_model.input)
    
    def get_training_input(self,batch_size):
        inp = [self.input_model.get_training_batch(batch_size)]
        if self.style_model is not None:
            inp.append(self.style_model.get_training_batch(batch_size))
        if self.noise_model is not None:
            inp.append(self.noise_model.get_training_batch(batch_size))
        return inp
    
    def get_validation_input(self,batch_size):
        inp = [self.input_model.get_validation_batch(batch_size)]
        if self.style_model is not None:
            inp.append(self.style_model.get_validation_batch(batch_size))
        if self.noise_model is not None:
            inp.append(self.noise_model.get_validation_batch(batch_size))
        return inp
    
    def build(self,print_summary=True):
        out = self.input_model.model
        for layer_config in self.gen_layers:            
            self.layer_sizes.append(list(filter(None,out.shape)))
            out = self.generator_block(out,layer_config)
        self.functional_model = out
        gen_model = Model(inputs=self.input,outputs=out,name="Generator")
        gen_model.compile(optimizer=self.gen_optimizer,
                          loss=self.loss_function,
                          metrics=self.metrics)
        if print_summary:
            gen_model.summary()
        return gen_model

    def generator_block(self,input_tensor,config: GenLayerConfig):
        out = input_tensor
        out = UpSampling2D(interpolation='bilinear')(out) if config.upsampling else out
        for i in range(config.convolutions):
            if config.transpose:
                out = Conv2DTranspose(config.filters,config.kernel_size,config.strides,
                                      padding='same',kernel_regularizer=self.kernel_regularizer.get(), 
                                      kernel_initializer = self.kernel_initializer)(out)
            else:
                out = Conv2D(config.filters,config.kernel_size,
                             padding='same',kernel_regularizer=self.kernel_regularizer.get(), 
                             kernel_initializer = self.kernel_initializer)(out)
            
            if self.noise_model is not None and config.noise:
                out = self.noise_model.add(out)
            
            if self.style_model is not None and config.style:
                beta = Dense(config.filters,bias_initializer='ones')(self.style_model.model)
                gamma = Dense(config.filters,bias_initializer='zeros')(self.style_model.model)
                out = AdaptiveInstanceNormalization()([out,beta,gamma])
            else:
                out = self.normalization.get()(out)
            out =  config.activation.get(out.shape)(out)
        return out
