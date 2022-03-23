from typing import List

import tensorflow as tf
from config.TrainingConfig import GanTrainingConfig, GanTrainingResult
from models.Discriminator import Discriminator
from models.Generator import Generator
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
            gen_images, gen_views = self.get_gen_output(gen_input,training=True)
            gen_results, disc_views = self.get_disc_output(gen_images, training=True)
            g_loss = self.gen_loss_function(tf.ones_like(gen_results)*self.gen_label,gen_results)
            metrics = []
            
            for metric in self.g_metrics:
                metric.update_state(gen_results)
                metrics.append(metric.result())
            
            gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return GanTrainingResult(g_loss,metrics)

    def train_discriminator(self, disc_input, gen_input):
        with tf.GradientTape() as disc_tape:
            gen_images,view_images = self.get_gen_output(gen_input,training=True)
            gen_results, disc_gen_views = self.get_disc_output(gen_images, training=True)
            real_results, disc_real_views = self.get_disc_output(disc_input, training=True)
            real_loss = self.disc_loss_function(tf.ones_like(real_results)*self.real_label,real_results)
            fake_loss = self.disc_loss_function(tf.ones_like(gen_results)*self.fake_label,gen_results)
            d_loss = real_loss + fake_loss 
            metrics = []
            
            for metric in self.d_metrics:
                metric.update_state(real_results)
                metrics.append(metric.result())
            
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return GanTrainingResult(d_loss,metrics)
