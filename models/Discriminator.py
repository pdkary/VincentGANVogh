from typing import List, Tuple

from config.GanConfig import DiscConvLayerConfig
from inputs.GanInput import GanInput
from layers.CallableConfig import (ActivationConfig, NoneCallable,
                                   RegularizationConfig)
from tensorflow.keras.losses import Loss
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.eager.monitoring import Metric

from models.ConvolutionalModel import ConvolutionalModel
from models.DenseModel import DenseModel


class Discriminator():
    def __init__(self,
                 real_image_input: GanInput,
                 disc_conv_layers: List[DiscConvLayerConfig],
                 disc_dense_layers: List[int],
                 optimizer: Optimizer,
                 loss_function: Loss,
                 metrics: List[Metric] = [],
                 minibatch_size: int = 0,
                 dropout_rate: float = 0.0,
                 activation: ActivationConfig = NoneCallable,
                 kernel_regularizer: RegularizationConfig = NoneCallable,
                 kernel_initializer: str = "glorot_uniform"):
        
        self.conv_model = ConvolutionalModel(gan_input=real_image_input,
                                             conv_layers=disc_conv_layers,
                                             kernel_regularizer=kernel_regularizer,
                                             kernel_initializer=kernel_initializer)

        self.conv_out = self.conv_model.build()
        
        self.dense_model = DenseModel(dense_input=self.conv_out,
                                      dense_layers=disc_dense_layers,
                                      activation=activation,
                                      minibatch_size=minibatch_size,
                                      dropout_rate=dropout_rate)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = [m() for m in metrics]
        self.metric_labels = ["D_" + str(m.name) for m in self.metrics]
        self.tracked_layers = self.conv_model.tracked_layers

    def build(self):
        self.functional_model = self.dense_model.build()
        disc_model = Model(inputs=self.conv_model.inputs, outputs=self.functional_model, name="Discriminator")
        disc_model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=self.metrics)
        disc_model.summary()
        return disc_model
