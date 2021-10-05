from typing import List, Tuple

from config.GanConfig import GenLayerConfig
from inputs.GanInput import GanInput
from layers.AdaptiveInstanceNormalization import AdaINConfig
from layers.CallableConfig import NoneCallable, NormalizationConfig, RegularizationConfig
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, UpSampling2D

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.metrics import Metric
from models.ConvolutionalModel import ConvolutionalModel

from models.StyleModel import LatentStyleModel


class Generator(ConvolutionalModel):
    def __init__(self,
                 gan_input: GanInput,
                 gen_layers: List[GenLayerConfig],
                 optimizer: Optimizer,
                 loss_function: Loss,
                 style_model: LatentStyleModel = None,
                 metrics: List[Metric] = [],
                 normalization: NormalizationConfig = NoneCallable,
                 kernel_regularizer:RegularizationConfig = NoneCallable,
                 kernel_initializer:str = "glorot_uniform"):
        super().__init__(gan_input,gen_layers,style_model,kernel_regularizer,kernel_initializer)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = [m() for m in metrics]
        self.metric_labels = ["G_" + str(m.name) for m in self.metrics]
        self.style_model = style_model
        self.normalization = normalization

    def build(self,print_summary=True):
        self.functional_model = ConvolutionalModel.build(self)
        gen_model = Model(inputs=self.inputs,outputs=self.functional_model)
        gen_model.compile(optimizer=self.optimizer,
                          loss=self.loss_function,
                          metrics=self.metrics)
        if print_summary:
            gen_model.summary()
        return gen_model

    def get_training_batch(self,batch_size):
        if self.style_model is not None and isinstance(self.style_model.dense_input,GanInput):
            return [self.gan_input.get_training_batch(batch_size),
                    self.style_model.dense_input.get_training_batch(batch_size)]
        else:
            return [self.gan_input.get_training_batch(batch_size)]

    def get_validation_batch(self,batch_size):
        if self.style_model is not None and isinstance(self.style_model.dense_input,GanInput):
            return [self.gan_input.get_validation_batch(batch_size),
                    self.style_model.dense_input.get_validation_batch(batch_size)]
        else:
            return [self.gan_input.get_validation_batch(batch_size)]
