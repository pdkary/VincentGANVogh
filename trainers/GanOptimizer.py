import numpy as np
import tensorflow as tf
from models.GanInput import RealImageInput
from typing import List
from keras_tuner import BayesianOptimization
from numpy.lib.function_base import median
from config.TrainingConfig import DataConfig, GanTrainingConfig
from models.Generator.HyperGAN import HyperDiscriminator, HyperGAN

class GanOptimizer(GanTrainingConfig):
    def __init__(self,
                 gan_training_config: GanTrainingConfig,
                 data_config:         DataConfig):
        super().__init__(**gan_training_config.__dict__)
        self.discriminator = HyperDiscriminator("hyper_discriminator",True)
        self.batch_size = data_config.batch_size
        self.preview_size = data_config.preview_cols*data_config.preview_rows
        self.generator = HyperGAN("hyper_generator",True,self.discriminator,self.batch_size,self.preview_size)
        self.image_source: RealImageInput = RealImageInput(data_config)
        self.image_source.load()
        self.model_output_path = data_config.data_path + "/models"

        self.gen_tuner = BayesianOptimization(
            self.generator,
            objective="accuracy",
            max_trials=5)
        
        self.disc_tuner = BayesianOptimization(
            self.discriminator,
            objective="accuracy",
            max_trials=5)

    def tune_on_batch(self,epoch):
        gen_input = self.generator.G.get_input()
        gen_images = self.generator.Gmodel(gen_input,training=False)
        real_images = self.image_source.get_batch()
        
        self.gen_tuner.search(gen_input,tf.ones_like(shape=(self.batch_size)),epochs=5)
        self.disc_tuner.search(real_images,tf.ones(shape=(self.batch_size)),epochs=5)
        self.disc_tuner.search(gen_images,tf.zeros(shape=(self.batch_size)),epochs=5)
        
        preview_seed = self.generator.G.get_input(training=False)
        generated_images = np.array(self.generator.Gmodel.predict(preview_seed))
        self.image_source.save(epoch,generated_images)
    
    def tune_n_batches(self,n):
        for i in range(n):
            self.tune_on_batch(i)
        filename = self.image_source.data_helper.model_name + "%d"%((i+1)*n)
        self.generator.Gmodel.save(self.model_output_path + filename)