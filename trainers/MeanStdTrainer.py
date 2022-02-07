from operator import ge
from typing import List
from cv2 import mean

import tensorflow as tf
import numpy as np
from config.GanConfig import TrackedLayerConfig
from config.TrainingConfig import GanTrainingConfig, GanTrainingResult
from models.Discriminator import Discriminator
from models.Generator import Generator
from trainers.AbstractTrainer import AbstractTrainer

def flatten(arr: List):
    return [x for y in arr for x in y]

class MeanStdTrainer(AbstractTrainer):
    def __init__(self, 
                 generator: Generator,
                 discriminator: Discriminator,
                 gan_training_config: GanTrainingConfig):
        super().__init__(generator, discriminator, gan_training_config)
        assert(generator.has_tracked_layers)
        assert(discriminator.has_tracked_layers)

    def train_generator(self,source_input, gen_input):
        with tf.GradientTape() as gen_tape:
            G_OUT = self.generator(gen_input,training=True)
            gen_images = G_OUT[0]
            gen_tracked_layers: List[TrackedLayerConfig] = G_OUT[1:]
            D_OUT = self.discriminator(gen_images, training=False)
            disc_results,disc_tracked_layers = D_OUT[0],D_OUT[1:]
            
            content_loss = self.gen_loss_function(self.gen_label, disc_results)
            g_mid = len(gen_tracked_layers)//2
            d_mid = len(disc_tracked_layers)//2
            g_means,g_stds = gen_tracked_layers[:g_mid],gen_tracked_layers[g_mid:]
            d_means,d_stds = disc_tracked_layers[:d_mid],disc_tracked_layers[d_mid:]
            mean_loss = self.gen_loss_function(d_means,g_means)
            std_loss = self.disc_loss_function(d_stds,g_stds)

            g_loss = content_loss + mean_loss + std_loss

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
            gen_images = self.generator(gen_input,training=False)[0]
            disc_gen_out = self.discriminator(gen_images, training=True)[0]

            disc_real_out = self.discriminator(disc_input, training=True)[0]
            
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
