from typing import List, Tuple

from config.GanConfig import GenLayerConfig
from inputs.GanInput import GanInput
from layers.AdaptiveInstanceNormalization import AdaINConfig
from layers.CallableConfig import NoneCallable, NormalizationConfig, RegularizationConfig
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, UpSampling2D

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from models.ConvolutionalModel import ConvolutionalModel

from models.DenseModel import LatentSpaceModel


class Generator():
    def __init__(self,
                 gan_input: GanInput,
                 gen_layers: List[GenLayerConfig],
                 style_model: LatentSpaceModel = None,
                 kernel_regularizer:RegularizationConfig = NoneCallable,
                 kernel_initializer:str = "glorot_uniform"):

        self.CM = ConvolutionalModel(gan_input=gan_input,
                                             conv_layers=gen_layers,
                                             style_model=style_model,
                                             kernel_regularizer=kernel_regularizer,
                                             kernel_initializer=kernel_initializer)

        self.functional_model = self.CM.build()
        self.tracked_layers = self.CM.tracked_layers
        self.style_model = style_model

    def get_training_batch(self,batch_size):
        b = [self.CM.gan_input.get_training_batch(batch_size)]
        if self.style_model is not None and isinstance(self.style_model,LatentSpaceModel):
            b.append(self.style_model.input.get_training_batch(batch_size))
        return b

    def get_validation_batch(self,batch_size):
        b = [self.CM.gan_input.get_validation_batch(batch_size)]
        if self.style_model is not None and isinstance(self.style_model,LatentSpaceModel):
            b.append(self.style_model.input.get_validation_batch(batch_size))
        return b
