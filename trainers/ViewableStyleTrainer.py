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
            gls,glm = self.G.tracked_layers[i]
            dls,dlm = self.D.tracked_layers[i]
            print("\tgen_style_std: ",gls.name,glm.shape)
            print("\tgen_style_mean: ",gls.name,glm.shape)
            print("\tdisc_style_std_%d: "%e,dls.name,dls.shape)
            print("\tdisc_style_mean_%d: "%e,dlm.name,dlm.shape)
            self.gen_style_layers.append(gls)
            self.gen_style_layers.append(glm)
            self.disc_style_layers.append(dls)
            self.disc_style_layers.append(dlm)

        self.gen_viewing_layers = self.G.viewing_layers
        self.disc_viewing_layers = self.D.viewing_layers
        print("\ngen_viewing_layers:")
        print([x.name for x in self.gen_viewing_layers])
        print([x.shape for x in self.gen_viewing_layers])

        self.style_end_index = 1 + len(self.gen_style_layers)
        
        self.generator = Model(inputs=GI,outputs=[GO,*self.gen_style_layers,*self.gen_viewing_layers])
        self.discriminator = Model(inputs=DI,outputs=[DO,*self.disc_style_layers,*self.disc_viewing_layers])

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
        self.g_metric_labels = ["G_Style_loss"] + self.g_metric_labels
        self.plot_labels = ["G_Loss","D_Loss",*self.g_metric_labels,*self.d_metric_labels]
    
    def save(self,epoch):
        preview_seed = self.G.get_validation_batch(self.preview_size)
        generated_images = np.array(self.generator.predict(preview_seed)[0])
        self.D.gan_input.save(epoch, generated_images, self.preview_rows, self.preview_cols, self.preview_margin)
        
    def get_all_style_loss(self,content_std_arr,content_mean_arr,style_std_arr,style_mean_arr):
        src_2_dest = list(zip(content_std_arr,content_mean_arr,style_std_arr,style_mean_arr))
        return [self.get_style_loss(cs,cm,ss,sm) for cs,cm,ss,sm in src_2_dest]
    
    def get_style_loss(self,content_std,content_mean,s_std,s_mean):
        std_error = self.style_loss_function(s_std,content_std)
        mean_error = self.style_loss_function(s_mean,content_mean)
        return [std_error,mean_error]

    def train_generator(self,source_input, gen_input):
        with tf.GradientTape() as gen_tape:
            gen_out = self.generator(gen_input,training=True)
            gen_images, gen_style, gen_view = gen_out[0],gen_out[1:self.style_end_index],gen_out[self.style_end_index:]
            gen_style_std,gen_style_mean = gen_style[0::2],gen_style[1::2]
            
            disc_out = self.discriminator(gen_images, training=False)
            disc_results,disc_style,disc_view = disc_out[0],disc_out[1:self.style_end_index],disc_out[self.style_end_index:]
            disc_style_std,disc_style_mean = disc_style[0::2],disc_style[1::2]

            content_loss = self.gen_loss_function(self.gen_label, disc_results)
            style_losses = self.get_all_style_loss(gen_style_std,gen_style_mean,disc_style_std,disc_style_mean) if len(self.matched_keys) > 0 else [] 
            
            vs = len(gen_view)
            view_losses = [self.style_loss_function(disc_view[i],gen_view[vs-i-1]) for i in range(vs)]            
            
            g_loss = [content_loss,*style_losses,*view_losses]
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
            gen_images, gen_style, gen_view = gen_out[0], gen_out[1:self.style_end_index], gen_out[self.style_end_index:]
            
            disc_gen_out = self.discriminator(gen_images, training=True)
            disc_gen_result,disc_gen_style,disc_gen_view = disc_gen_out[0],disc_gen_out[1:self.style_end_index],disc_gen_out[self.style_end_index:]

            disc_real_out = self.discriminator(disc_input, training=True)
            disc_real_result, disc_real_style, disc_real_view = disc_real_out[0],disc_real_out[1:self.style_end_index],disc_real_out[self.style_end_index:]
            
            content_loss = self.disc_loss_function(self.fake_label, disc_gen_result)
            content_loss += self.disc_loss_function(self.real_label, disc_real_result)
            
            style_losses = [tf.zeros_like(x) for x in gen_style]
            view_losses = [tf.zeros_like(x) for x in disc_real_view]            
            
            d_loss = [content_loss,*style_losses,*view_losses]
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
