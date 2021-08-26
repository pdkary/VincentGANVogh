


import tensorflow as tf
from tensorflow.python import training
from models.GanInput import RealImageInput
from typing import List
from keras_tuner import BayesianOptimization
from numpy.lib.function_base import median
from config.TrainingConfig import DataConfig, GanTrainingConfig
from models.Generator.HyperGAN import HyperDiscriminator, HyperGenerator


class GanOptimizer(GanTrainingConfig):
    def __init__(self,
                 gan_training_config: GanTrainingConfig,
                 data_config:         DataConfig):
        super().__init__(**gan_training_config.__dict__)
        self.discriminator = HyperDiscriminator("hyper_discriminator",True)
        self.batch_size = data_config.batch_size
        self.preview_size = data_config.preview_cols*data_config.preview_rows
        self.generator = HyperGenerator("hyper_generator",True,4,24)
        self.image_source: RealImageInput = RealImageInput(data_config)
        self.image_source.load()
        
        self.gen_tuner = BayesianOptimization(
            self.generator,
            objective="val_accuracy",
            max_trials=5)
        
        self.disc_tuner = BayesianOptimization(
            self.discriminator,
            objective="val_accuracy",
            max_trials=5)

    def tune(self):
        gen_input = self.generator.G.get_input()
        gen_images = self.generator.G(gen_input,training=False)
        
        disc_gens = self.discriminator.D(gen_images,training=False)
        disc_reals = self.discriminator.D(self.image_source.get_batch())
        
        self.gen_tuner.search(tf.ones_like(disc_gens),disc_gens,epochs=2)
        self.disc_tuner.search(tf.ones_like(disc_reals),disc_reals,epochs=2)
        self.disc_tuner.search(tf.zeros_like(disc_gens),disc_gens,epochs=2)
        