from abc import ABC, abstractmethod
from typing import Tuple

import tensorflow as tf
from config.GanConfig import ActivationConfig, RegularizationConfig
from layers.GanInput import RealImageInput
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Activation


class StyleModelBase(ABC):
    def __init__(self,
                 input_shape: Tuple,
                 style_layers: int,
                 style_layer_size: int,
                 activation: ActivationConfig):
        self.input_shape = input_shape
        self.style_layers = style_layers
        self.style_layer_size = style_layer_size
        self.activation = activation
        self.input = Input(shape=input_shape, name="style_model_input")

    @abstractmethod
    def get_training_batch(self, batch_size: int):
        pass

    @abstractmethod
    def get_validation_batch(self, batch_size: int):
        pass


class LatentStyleModel(StyleModelBase):
    def __init__(self,
                 input_shape: Tuple,
                 style_layers: int,
                 style_layer_size: int,
                 activation: ActivationConfig):
        super().__init__(input_shape, style_layers, style_layer_size, activation)

        model = self.input
        for i in range(self.style_layers):
            model = Dense(self.style_layer_size)(model)
            model = self.activation.get()(model)
        self.model = model

    def get_training_batch(self, batch_size: int):
        return tf.random.normal(shape=(batch_size, self.input_shape), dtype=tf.float32)

    def get_validation_batch(self, batch_size: int):
        return self.get_training_batch(batch_size)


class ImageStyleModel(StyleModelBase):
    def __init__(self,
                 real_image_input: RealImageInput,
                 filters: int,
                 convolutions: int,
                 kernel_size: int,
                 style_layers: int,
                 style_layer_size: int,
                 activation: ActivationConfig,
                 conv_activation: ActivationConfig,
                 kernel_regularizer: RegularizationConfig,
                 kernel_initializer: str = "glorot_uniform",
                 downsample_factor: int = 1):
        self.image_source = real_image_input
        self.downsample_factor = (downsample_factor, downsample_factor)
        super().__init__(real_image_input.input_shape,
                         style_layers, style_layer_size, activation)

        model = MaxPooling2D(self.downsample_factor)(self.input)
        for i in range(convolutions):
            model = Conv2D(filters, kernel_size, padding="same",
                           kernel_regularizer=kernel_regularizer.get(),
                           kernel_initializer=kernel_initializer)(model)
            model = conv_activation.get()(model)

        model = Flatten()(model) if self.style_layers > 0 else model

        for i in range(self.style_layers):
            model = Dense(self.style_layer_size)(model)
            model = self.activation.get()(model)
        self.model = model

    def get_training_batch(self, batch_size):
        return self.image_source.get_training_batch(batch_size)

    def get_validation_batch(self, batch_size):
        return self.image_source.get_validation_batch(batch_size)
