from helpers.DataHelper import shape_to_key
from typing import List, Tuple

from tensorflow.python.eager.monitoring import Metric

from config.GanConfig import DiscConvLayerConfig, DiscDenseLayerConfig
from layers.CallableConfig import ActivationConfig, NoneCallable, RegularizationConfig
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer
from third_party_layers.MinibatchDiscrimination import MinibatchDiscrimination


class Discriminator():
    def __init__(self,
                 img_shape: Tuple[int, int, int],
                 disc_conv_layers: List[DiscConvLayerConfig],
                 disc_dense_layers: List[DiscDenseLayerConfig],
                 disc_optimizer: Optimizer,
                 loss_function: Loss,
                 metrics: List[Metric] = [],
                 minibatch_size: int = 0,
                 kernel_regularizer: RegularizationConfig = NoneCallable,
                 kernel_initializer: str = "glorot_uniform"):
        self.img_shape = img_shape
        self.disc_conv_layers = disc_conv_layers
        self.disc_dense_layers = disc_dense_layers
        self.minibatch_size = minibatch_size
        self.minibatch = minibatch_size > 0
        self.disc_optimizer = disc_optimizer
        self.loss_function = loss_function
        self.metrics = [m() for m in metrics]
        self.metric_labels = ["D_" + str(m.name) for m in self.metrics]
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.output_dim = self.disc_dense_layers[-1].size
        self.layer_sizes = [self.img_shape]
        self.input = Input(shape=self.img_shape, name="discriminator_input")

        self.tracked_layers = []

    def build(self):
        out = self.input
        for layer_config in self.disc_conv_layers:
            self.layer_sizes.append(list(filter(None,out.shape)))
            out = self.disc_conv_block(out, layer_config)

        out = Flatten()(out)

        for layer_config in self.disc_dense_layers:
            self.layer_sizes.append(list(filter(None,out.shape)))
            out = self.disc_dense_block(out, layer_config)

        self.functional_model = out
        disc_model = Model(inputs=self.input, outputs=out, name="Discriminator")
        disc_model.compile(optimizer=self.disc_optimizer,
                           loss=self.loss_function,
                           metrics=self.metrics)
        disc_model.summary()
        return disc_model

    def disc_dense_block(self, input_tensor, config: DiscDenseLayerConfig):
        out_db = MinibatchDiscrimination(self.minibatch_size, self.img_shape[-1])(input_tensor) if self.minibatch else input_tensor
        out_db = Dense(config.size)(out_db)
        out_db = Dropout(config.dropout_rate)(out_db) if config.dropout_rate > 0 else out_db
        out_db = config.activation.get(config.size)(out_db)
        return out_db

    def disc_conv_block(self, input_tensor, config: DiscConvLayerConfig):
        out_cb = input_tensor
        out_cb = MaxPooling2D()(out_cb) if config.downsampling else out_cb
        for i in range(config.convolutions):
            out_cb = Conv2D(config.filters, config.kernel_size, padding="same", 
                            kernel_regularizer=self.kernel_regularizer.get(), 
                            kernel_initializer=self.kernel_initializer,use_bias=False)(out_cb)
            out_cb = config.normalization.get()(out_cb)
            act_name = "_".join(["activation", shape_to_key(out_cb.shape), str(i)])
            out_cb = config.activation.get(name=act_name)(out_cb)
            self.tracked_layers.append(out_cb)
            out_cb = Dropout(config.dropout_rate)(out_cb) if config.dropout_rate > 0 else out_cb
        return out_cb
