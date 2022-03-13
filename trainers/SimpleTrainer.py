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
            disc_results, disc_views = self.get_disc_output(gen_images, training=True)
            content_loss = self.gen_loss_function(self.gen_label, disc_results)

            g_loss = content_loss
            metrics = []
            
            for metric in self.g_metrics:
                if metric.name == "mean":
                    metric.update_state(disc_results)
                else:
                    metric.update_state(self.gen_label,disc_results)
                metrics.append(metric.result())
            
            gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return GanTrainingResult(g_loss,metrics)

    def train_discriminator(self, disc_input, gen_input):
        with tf.GradientTape() as disc_tape:
            gen_images,view_images = self.get_gen_output(gen_input,training=True)
            disc_gen_out, disc_gen_views = self.get_disc_output(gen_images, training=True)
            disc_real_out, disc_real_views = self.get_disc_output(disc_input, training=True)
            
            content_loss =  self.disc_loss_function(self.fake_label, disc_gen_out) 
            content_loss += self.disc_loss_function(self.real_label, disc_real_out)

            d_loss = content_loss
            metrics = []
            
            for metric in self.d_metrics:
                if metric.name == "mean":
                    metric.update_state(disc_real_out)
                else:
                    metric.update_state(self.real_label,disc_real_out)
                metrics.append(metric.result())
            
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return GanTrainingResult(d_loss,metrics)
