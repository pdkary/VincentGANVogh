from abc import ABC, abstractmethod
from layers.GanInput import RealImageInput
from typing import Tuple

import numpy as np
import tensorflow as tf
from config.GanConfig import ActivationConfig, RegularizationConfig
from layers.AdaptiveAdd import AdaptiveAdd
from tensorflow.keras.layers import Conv2D, Cropping2D, Input


class NoiseModelBase(ABC):
    def __init__(self,
                 noise_image_size: Tuple[int, int, int],
                 activation: ActivationConfig,
                 kernel_regularizer: RegularizationConfig,
                 kernel_initializer: str = "glorot_normal",
                 kernel_size: int = 1,
                 max_std_dev: float = 1.0):
        self.noise_image_size = noise_image_size
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.kernel_size = kernel_size
        self.max_std_dev = max_std_dev

        self.input = Input(shape=self.noise_image_size,name="noise_model_input")
        self.model = self.activation.get()(self.input)

    def add(self, input_tensor):
        n_size = self.model.shape[1]
        i_size = input_tensor.shape[1]
        noise = Cropping2D((n_size-i_size)//2)(self.model)
        noise = Conv2D(input_tensor.shape[-1],
                       self.kernel_size,
                       padding='same',
                       kernel_regularizer=self.kernel_regularizer.get(),
                       kernel_initializer=self.kernel_initializer)(noise)
        return AdaptiveAdd()([input_tensor, noise])

    @abstractmethod
    def get_training_batch(self, batch_size: int):
        pass
    
    @abstractmethod
    def get_validation_batch(self, batch_size: int):
        pass


class LatentNoiseModel(NoiseModelBase):
    def __init__(self, 
                 noise_image_size: Tuple[int, int, int], 
                 activation: ActivationConfig, 
                 kernel_regularizer: RegularizationConfig, 
                 kernel_initializer: str = "glorot_uniform", 
                 kernel_size: int = 1, 
                 max_std_dev: float = 1.0):
        super().__init__(noise_image_size, activation, kernel_regularizer,kernel_initializer, kernel_size, max_std_dev)
        
    def get_training_batch(self, batch_size: int):
        noise_batch = np.full((batch_size, *self.noise_image_size), 0.0, dtype=np.float32)
        for i in range(batch_size):
            noise_batch[i] = tf.random.normal(shape=self.noise_image_size, stddev=self.max_std_dev)
        return noise_batch
    
    def get_validation_batch(self, batch_size: int):
        return self.get_training_batch(batch_size)


class ConstantNoiseModel(NoiseModelBase):
    def __init__(self, 
                 noise_image_size: Tuple[int, int, int], 
                 activation: ActivationConfig, 
                 kernel_regularizer: RegularizationConfig, 
                 kernel_initializer: str = "glorot_uniform", 
                 kernel_size: int = 1, 
                 max_std_dev: float = 1.0):
        super().__init__(noise_image_size, activation, kernel_regularizer,kernel_initializer, kernel_size, max_std_dev)
        self.constant = tf.random.normal(shape=self.noise_image_size, stddev=self.max_std_dev)

    def get_training_batch(self, batch_size: int):
        noise_batch = np.full(
            (batch_size, *self.noise_image_size), 0.0, dtype=np.float32)
        for i in range(batch_size):
            noise_batch[i] = self.constant
        return noise_batch
    
    def get_validation_batch(self, batch_size: int):
        return self.get_training_batch(batch_size)


class ImageNoiseModel(NoiseModelBase):
    def __init__(self, 
                 image_source: RealImageInput, 
                 activation: ActivationConfig, 
                 kernel_regularizer: RegularizationConfig, 
                 kernel_initializer: str = "glorot_uniform", 
                 kernel_size: int = 1, 
                 max_std_dev: float = 1.0):
        self.image_source = image_source
        super().__init__(image_source.image_shape, activation, kernel_regularizer, kernel_initializer, kernel_size, max_std_dev)
        
    def get_training_batch(self, batch_size):
        return self.image_source.get_training_batch(batch_size)

    def get_validation_batch(self, batch_size):
        return self.image_source.get_validation_batch(batch_size)