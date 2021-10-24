from typing import List

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.python.keras.backend import dtype, zeros_like
from config.TrainingConfig import GanTrainingConfig
from models.Discriminator import Discriminator
from models.Generator import Generator
from tensorflow.keras.models import Model
from trainers.AbstractTrainer import AbstractTrainer

def flatten(arr: List):
    return [x for y in arr for x in y]

class ViewableTrainer(AbstractTrainer):
    def __init__(self, 
                 generator: Generator,
                 discriminator: Discriminator,
                 gan_training_config: GanTrainingConfig):
        super().__init__(generator, discriminator, gan_training_config)

    def compile(self):
        GI,GO = self.G.input,self.G.build()
        DI,DO = self.D.input,self.D.build()

        self.gen_viewing_layers = self.G.viewing_layers
        self.disc_viewing_layers = self.D.viewing_layers
        print("\ngen_viewing_layers:")
        print([x.name for x in self.gen_viewing_layers])
        print([x.shape for x in self.gen_viewing_layers])
        
        self.generator = Model(inputs=GI,outputs=[GO,*self.gen_viewing_layers])
        self.discriminator = Model(inputs=DI,outputs=[DO,*self.disc_viewing_layers])

        self.generator.compile(optimizer=self.gen_optimizer,
                               loss=self.gen_loss_function,
                               metrics=self.g_metrics)
        self.discriminator.compile(optimizer=self.disc_optimizer,
                                   loss=self.disc_loss_function,
                                   metrics=self.d_metrics)
        self.generator.summary()
        self.discriminator.summary()
    
    def save_images(self,name):
        preview_seed = self.G.get_validation_batch(self.preview_size)
        gen_out = self.generator.predict(preview_seed)
        gen_images, gen_views = gen_out[0],gen_out[1:]
        self.D.gan_input.save_viewed(name,gen_images,gen_views,self.preview_margin)

    def train_generator(self,source_input, gen_input):
        with tf.GradientTape() as gen_tape:
            gen_out = self.generator(gen_input,training=True)
            gen_images, gen_view = gen_out[0],gen_out[1:]
            
            disc_out = self.discriminator(gen_images, training=False)
            disc_results,disc_view = disc_out[0],disc_out[1:]

            content_loss = self.gen_loss_function(self.gen_label, disc_results)
            content_loss += self.gen_loss_function(source_input,gen_images)
            view_losses = [tf.zeros_like(x) for x in gen_view]

            g_loss = [content_loss,*view_losses]
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
            gen_out = self.generator(gen_input,training=False)
            gen_images, gen_view = gen_out[0], gen_out[1:]

            disc_gen_out = self.discriminator(gen_images, training=True)
            disc_gen_result,disc_gen_view = disc_gen_out[0],disc_gen_out[1:]
           
            disc_real_out = self.discriminator(disc_input, training=True)
            disc_real_result,disc_real_view = disc_real_out[0],disc_real_out[1:]
            
            content_loss =  self.disc_loss_function(self.fake_label, disc_gen_result) 
            content_loss += self.disc_loss_function(self.real_label, disc_real_result)
            view_losses = [tf.zeros_like(x) for x in disc_real_view] 

            d_loss = [content_loss,*view_losses]
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
