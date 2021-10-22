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

class ViewableStyleTrainer(AbstractTrainer):
    def __init__(self, 
                 generator: Generator,
                 discriminator: Discriminator,
                 gan_training_config: GanTrainingConfig):
        super().__init__(generator, discriminator, gan_training_config)

    def compile(self):
        GI,GO = self.G.input,self.G.build()
        DI,DO = self.D.input,self.D.build()
        self.matched_keys = [g for g in self.G.tracked_layers.keys() if g in self.D.tracked_layers]

        print("COMPILING")
        self.disc_style_layers = []
        self.gen_style_layers = []
        for e,i in enumerate(self.matched_keys):
            print("\nLAYER %d-----"%e)
            gl = self.G.tracked_layers[i]
            dl = self.D.tracked_layers[i]
            print("\tgen_style_std: ",gl[0].name,gl[0].shape)
            print("\tgen_style_mean: ",gl[1].name,gl[1].shape)
            print("\tdisc_style_std_%d: "%e,dl[0].name,dl[0].shape)
            print("\tdisc_style_mean_%d: "%e,dl[1].name,dl[1].shape)
            self.gen_style_layers.append(gl)
            self.disc_style_layers.append(dl)

        self.gen_viewing_layers = self.G.viewing_layers
        print("\ngen_viewing_layers:")
        print([x.name for x in self.gen_viewing_layers])
        print([x.shape for x in self.gen_viewing_layers])
        
        self.generator = Model(inputs=GI,outputs=[GO,self.gen_style_layers,self.gen_viewing_layers])
        self.discriminator = Model(inputs=DI,outputs=[DO,self.disc_style_layers])

        self.generator.compile(optimizer=self.gen_optimizer,
                               loss=self.gen_loss_function,
                               metrics=self.g_metrics)
        self.discriminator.compile(optimizer=self.disc_optimizer,
                                   loss=self.disc_loss_function,
                                   metrics=self.d_metrics)
        self.generator.summary()
        self.discriminator.summary()

        print("MATCHED LAYERS: ")
        print(self.matched_keys)            
        self.nil_disc_style_loss = [[tf.zeros_like(a,dtype=tf.float32),tf.zeros_like(b,dtype=tf.float32)] for a,b in self.disc_style_layers]
        print("nil_disc_style_loss: ",self.nil_disc_style_loss)
        self.g_metric_labels = ["G_Style_loss"] + self.g_metric_labels
        self.plot_labels = ["G_Loss","D_Loss",*self.g_metric_labels,*self.d_metric_labels]
    
    def save(self,epoch):
        preview_seed = self.G.get_validation_batch(self.preview_size)
        generated_images = np.array(self.generator.predict(preview_seed)[0])
        self.D.gan_input.save(epoch, generated_images, self.preview_rows, self.preview_cols, self.preview_margin)
        
    def get_deep_style_loss(self,content_std_arr,content_mean_arr,style_std_arr,style_mean_arr):
        src_2_dest = list(zip(content_std_arr,content_mean_arr,style_std_arr,style_mean_arr))
        return [self.get_style_loss(cs,cm,ss,sm) for cs,cm,ss,sm in src_2_dest]
    
    def get_style_loss(self,content_std,content_mean,s_std,s_mean):
        std_error = self.style_loss_function(s_std,content_std)
        mean_error = self.style_loss_function(s_mean,content_mean)
        return [std_error,mean_error]

    def train_generator(self,source_input, gen_input):
        with tf.GradientTape() as gen_tape:
            gen_out,gen_style_layers,gen_view_layers = self.generator(gen_input,training=True)
            gen_style_std = [x[0] for x in gen_style_layers]
            gen_style_mean = [x[1] for x in gen_style_layers]
            disc_out = self.discriminator(gen_out, training=False)
            
            if len(self.matched_keys) > 0:
                disc_results,disc_deep_layers = disc_out[0],disc_out[1:]
            else:
                disc_results,disc_deep_layers = disc_out,None
            
            content_loss = self.gen_loss_function(self.gen_label, disc_results)
            deep_style_losses = self.get_deep_style_loss(gen_style_std,gen_style_mean,disc_deep_layers) if len(self.matched_keys) > 0 else [] 
            
            total_loss = content_loss + self.style_loss_coeff*np.sum(deep_style_losses)
            g_loss = [total_loss,*deep_style_losses]
            out = [content_loss, np.sum(deep_style_losses)]
            
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

            if len(self.matched_keys) > 0:
                disc_real_results = self.discriminator(disc_input, training=True)[0]
                disc_gen_results = self.discriminator(gen_out[0], training=True)[0]
            else:
                disc_real_results = self.discriminator(disc_input, training=True)
                disc_gen_results = self.discriminator(gen_out, training=True)
            
            content_loss = self.disc_loss_function(self.real_label, disc_real_results)
            content_loss += self.disc_loss_function(self.fake_label, disc_gen_results)

            total_loss = content_loss
            d_loss = [total_loss, *self.nil_disc_style_loss]
            out = [content_loss]
            
            for metric in self.d_metrics:
                if metric.name == "mean":
                    metric.update_state(disc_real_results)
                else:
                    metric.update_state(self.real_label,disc_real_results)
                out.append(metric.result())
            
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return out
