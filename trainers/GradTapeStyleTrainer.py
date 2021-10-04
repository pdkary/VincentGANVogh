from typing import List

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from config.TrainingConfig import GanTrainingConfig
from layers.GanInput import RealImageInput
from models.Discriminator import Discriminator
from models.Generator import Generator
from tensorflow.keras.models import Model

from trainers.AbstractTrainer import AbstractTrainer

def flatten(arr: List):
    return [x for y in arr for x in y]

class GradTapeStyleTrainer(AbstractTrainer):
    def __init__(self, 
                 generator: Generator,
                 discriminator: Discriminator,
                 gan_training_config: GanTrainingConfig,
                 image_sources: List[RealImageInput]):
        super().__init__(generator, discriminator, gan_training_config, image_sources)

        self.matched_keys = [g for g in self.G.tracked_layers.keys() if g in self.D.tracked_layers]
        self.gen_deep_layers = flatten([self.G.tracked_layers[i] for i in self.matched_keys])
        self.disc_deep_layers = flatten([self.D.tracked_layers[i] for i in self.matched_keys])
        
        self.generator = Model(inputs=self.G.input,outputs=[self.G.functional_model,*self.gen_deep_layers])
        self.discriminator = Model(inputs=self.D.input,outputs=[self.D.functional_model,*self.disc_deep_layers])

        self.nil_disc_style_loss = tf.constant([0.0 for i in self.disc_deep_layers],dtype=tf.float32)

        self.G.metric_labels = ["G_Style_loss"] + self.G.metric_labels
        self.plot_labels = ["G_Loss","D_Loss",*self.G.metric_labels,*self.D.metric_labels]
    
    def save(self,epoch):
        preview_seed = self.G.get_validation_input(self.preview_size)
        generated_images = np.array(self.generator.predict(preview_seed)[0])
        self.image_sources[0].save(epoch, generated_images, self.preview_rows, self.preview_cols, self.preview_margin)
        
    def get_deep_style_loss(self,content_std,content_mean,style_src):
        src_2_dest = list(zip(content_std,content_mean,style_src))
        unflat_result = [self.get_style_loss(si,mu,s) for si,mu,s in src_2_dest]
        return [x for y in unflat_result for x in y]
    
    def get_style_loss(self,content_std,content_mean,style_img):
        s_mean = K.mean(style_img,[1,2],keepdims=True)
        s_std = K.std(style_img,[1,2],keepdims=True)
        mean_error = self.style_loss_function(s_mean,content_mean)
        std_error = self.style_loss_function(s_std,content_std)
        return [std_error,mean_error]
        
    def train_generator(self,source_input, gen_input):
        with tf.GradientTape() as gen_tape:
            gen_out = self.generator(gen_input,training=True)
            gen_images,gen_deep_std_layers,gen_deep_mean_layers = gen_out[0],gen_out[1::2],gen_out[2::2]

            disc_out = self.discriminator(gen_images, training=False)
            disc_results,disc_deep_layers = disc_out[0],disc_out[1:]
            
            content_loss = self.G.loss_function(self.gen_label, disc_results)
            deep_style_losses = self.get_deep_style_loss(gen_deep_std_layers,gen_deep_mean_layers,disc_deep_layers)
            
            total_loss = content_loss + self.style_loss_coeff*np.sum(deep_style_losses)
            g_loss = [total_loss,*deep_style_losses]
            out = [content_loss, np.sum(deep_style_losses)]
            
            for metric in self.G.metrics:
                if metric.name == "mean":
                    metric.update_state(disc_results)
                else:
                    metric.update_state(self.gen_label,disc_results)
                out.append(metric.result())
            
            gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.G.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return out

    def train_discriminator(self, disc_input, gen_input):
        with tf.GradientTape() as disc_tape:
            gen_out = self.generator(gen_input,training=False)[0]
            
            disc_real_results = self.discriminator(disc_input, training=True)[0]
            disc_gen_results = self.discriminator(gen_out, training=True)[0]
            
            content_loss = self.D.loss_function(self.real_label, disc_real_results)
            content_loss += self.D.loss_function(self.fake_label, disc_gen_results)

            total_loss = content_loss
            d_loss = [total_loss, *self.nil_disc_style_loss]
            out = [content_loss]
            
            for metric in self.D.metrics:
                if metric.name == "mean":
                    metric.update_state(disc_real_results)
                else:
                    metric.update_state(self.real_label,disc_real_results)
                out.append(metric.result())
            
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.D.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return out
