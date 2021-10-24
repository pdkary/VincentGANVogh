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

class StyleTrainer(AbstractTrainer):
    def __init__(self, 
                 generator: Generator,
                 discriminator: Discriminator,
                 gan_training_config: GanTrainingConfig):
        super().__init__(generator, discriminator, gan_training_config)
    
    def get_all_style_loss(self,content_std_arr,content_mean_arr,style_std_arr,style_mean_arr):
        src_2_dest = list(zip(content_std_arr,content_mean_arr,style_std_arr,style_mean_arr))
        return [self.get_style_loss(cs,cm,ss,sm) for cs,cm,ss,sm in src_2_dest]
    
    def get_style_loss(self,content_std,content_mean,s_std,s_mean):
        std_error = self.style_loss_coeff*self.style_loss_function(s_std,content_std)
        mean_error = self.style_loss_coeff*self.style_loss_function(s_mean,content_mean)
        return [std_error,mean_error]

    def compile(self):
        GI,GO = self.G.input,self.G.build()
        DI,DO = self.D.input,self.D.build()
        self.matched_keys = [g for g in self.G.tracked_layers.keys() if g in self.D.tracked_layers]

        self.disc_style_layers = []
        self.gen_style_layers = []
        for e,i in enumerate(self.matched_keys):
            gls,glm = self.G.tracked_layers[i]
            dls,dlm = self.D.tracked_layers[i]
            self.gen_style_layers.append(gls)
            self.gen_style_layers.append(glm)
            self.disc_style_layers.append(dls)
            self.disc_style_layers.append(dlm)

        self.generator = Model(inputs=GI,outputs=[GO,*self.gen_style_layers])
        self.discriminator = Model(inputs=DI,outputs=[DO,*self.disc_style_layers])

        self.generator.compile(optimizer=self.gen_optimizer,
                               loss=self.gen_loss_function,
                               metrics=self.g_metrics)
        self.discriminator.compile(optimizer=self.disc_optimizer,
                                   loss=self.disc_loss_function,
                                   metrics=self.d_metrics)
        self.generator.summary()
        self.discriminator.summary()

    def save(self,epoch):
        preview_seed = self.G.get_validation_batch(self.preview_size)
        generated_images = np.array(self.generator.predict(preview_seed)[0])
        self.D.gan_input.save(epoch, generated_images, self.preview_rows, self.preview_cols, self.preview_margin)
        
    def get_all_style_loss(self,content_std_arr,content_mean_arr,style_std_arr,style_mean_arr):
        src_2_dest = list(zip(content_std_arr,content_mean_arr,style_std_arr,style_mean_arr))
        return [self.get_style_loss(cs,cm,ss,sm) for cs,cm,ss,sm in src_2_dest]
    
    def get_style_loss(self,content_std,content_mean,s_std,s_mean):
        std_error = self.style_loss_coeff*self.style_loss_function(s_std,content_std)
        mean_error = self.style_loss_coeff*self.style_loss_function(s_mean,content_mean)
        return [std_error,mean_error]

    def train_generator(self,source_input, gen_input):
        with tf.GradientTape() as gen_tape:
            gen_out = self.generator(gen_input,training=True)
            gen_images, gen_style = gen_out[0],gen_out[1:]
            gen_style_std,gen_style_mean = gen_style[0::2],gen_style[1::2]
            
            disc_out = self.discriminator(gen_images, training=False)
            disc_results,disc_style = disc_out[0],disc_out[1:]
            disc_style_std,disc_style_mean = disc_style[0::2],disc_style[1::2]

            content_loss = self.gen_loss_function(self.gen_label, disc_results)
            content_loss += (1e-3)*self.gen_loss_function(source_input,gen_images)

            style_losses = self.get_all_style_loss(gen_style_std,gen_style_mean,disc_style_std,disc_style_mean) if len(self.matched_keys) > 0 else []

            g_loss = [content_loss,*style_losses]
            out = [content_loss, np.sum(style_losses)]
            
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
            if len(gen_out) > 1:
                gen_images, gen_style = gen_out[0], gen_out[1:]
            else:            
                gen_images, gen_style = gen_out[0], None

            disc_gen_out = self.discriminator(gen_images, training=True)
            if len(disc_gen_out) > 1:
                disc_gen_result,disc_gen_style = disc_gen_out[0],disc_gen_out[1:]
            else:            
                disc_gen_result,disc_gen_style = disc_gen_out[0],None

            disc_real_out = self.discriminator(disc_input, training=True)
            if len(disc_real_out) > 1:
                disc_real_result,disc_real_style = disc_real_out[0],disc_real_out[1:]
            else:            
                disc_real_result,disc_real_style = disc_real_out[0],None
            
            content_loss = self.disc_loss_function(self.fake_label, disc_gen_result) + self.disc_loss_function(self.real_label, disc_real_result)
            style_losses = [tf.zeros_like(x) for x in disc_real_style] 

            d_loss = [content_loss,*style_losses]
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
