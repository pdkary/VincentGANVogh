from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import tensorflow as tf
from layers.CallableConfig import ActivationConfig
from layers.GanInput import RealImageInput
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input


class NoiseModelBase(ABC):
    def __init__(self,
                 noise_image_size: Tuple[int, int, int],
                 activation: ActivationConfig,
                 kernel_size: int = 1,
                 max_std_dev: float = 1.0):
        self.noise_image_size = noise_image_size
        self.activation = activation
        self.kernel_size = kernel_size
        self.max_std_dev = max_std_dev

        self.input = Input(shape=self.noise_image_size,name="noise_model_input")
        model = self.activation.get()(self.input)
        self.model = Model(inputs=self.input,outputs=model)

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
                 kernel_size: int = 1, 
                 max_std_dev: float = 1.0):
        super().__init__(noise_image_size, activation, kernel_size, max_std_dev)
        
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
                 kernel_size: int = 1, 
                 max_std_dev: float = 1.0):
        super().__init__(noise_image_size, activation, kernel_size, max_std_dev)
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
                 kernel_size: int = 1, 
                 max_std_dev: float = 1.0):
        self.image_source = image_source
        super().__init__(image_source.image_shape, activation, kernel_size, max_std_dev)
        
    def get_training_batch(self, batch_size):
        return self.image_source.get_training_batch(batch_size)

    def get_validation_batch(self, batch_size):
        return self.image_source.get_validation_batch(batch_size)

