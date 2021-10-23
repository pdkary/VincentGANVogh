from typing import List

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from config.TrainingConfig import GanTrainingConfig
from models.Discriminator import Discriminator
from models.Generator import Generator
from tensorflow.keras.models import Model
from trainers.AbstractTrainer import AbstractTrainer

def flatten(arr: List):
    return [x for y in arr for x in y]

class SimpleTrainer(AbstractTrainer):
    def __init__(self, 
                 generator: Generator,
                 discriminator: Discriminator,
                 gan_training_config: GanTrainingConfig):
        super().__init__(generator, discriminator, gan_training_config)

    def train_generator(self,source_input, gen_input):
        with tf.GradientTape() as gen_tape:
            gen_images = self.generator(gen_input,training=True)
            disc_results = self.discriminator(gen_images, training=False)
            content_loss = self.gen_loss_function(self.gen_label, disc_results)

            g_loss = [content_loss]
            out = [content_loss]
            
            for metric in self.g_metrics:
                if metric.name == "mean":
                    metric.update_state(disc_results)
                else:
                    metric.update_state(self.gen_label,disc_results)
                out.append(metric.result())
            
            gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return out

    def train_discriminator(self, disc_input, gen_input):
        with tf.GradientTape() as disc_tape:
            gen_images = self.generator(gen_input,training=False)
            disc_gen_result = self.discriminator(gen_images, training=True)

            disc_real_out = self.discriminator(disc_input, training=True)
            disc_real_result, disc_real_style = disc_real_out[0],disc_real_out[1:]
            
            content_loss = self.disc_loss_function(self.fake_label, disc_gen_result) + self.disc_loss_function(self.real_label, disc_real_result)

            d_loss = [content_loss]
            out = [content_loss]
            
            for metric in self.d_metrics:
                if metric.name == "mean":
                    metric.update_state(disc_real_result)
                else:
                    metric.update_state(self.real_label,disc_real_result)
                out.append(metric.result())
            
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return out
