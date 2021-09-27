from os import name
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


class MatchedGanStyleTrainer(AbstractTrainer):
    def __init__(self, 
                 generator: Generator,
                 discriminator: Discriminator,
                 gan_training_config: GanTrainingConfig,
                 image_sources: List[RealImageInput]):
        super().__init__(generator, discriminator, gan_training_config, image_sources)
        g_tracked = [x for y in self.G.tracked_layers for x in y]

        g_tracked_std = [l for l in g_tracked if "std" in l.name]
        g_tracked_mean = [l for l in g_tracked if "mean" in l.name]
        g_shapes = [list(filter(None,l.shape))[0] for l in g_tracked_std]
        print(g_shapes)
        
        d_tracked = list(reversed(self.D.tracked_layers))
        d_shapes = [list(filter(None,l.shape))[-1]for l in d_tracked]
        print(d_shapes)

        match_indicies = [i for i,val in enumerate(d_shapes) if val == g_shapes[i]]
        self.matched_layers = [(g_tracked_std[i].name,g_tracked_mean[i].name,d_tracked[i].name) for i in match_indicies]
        print("MATCHING LAYERS: \n",self.matched_layers)

        self.gen_deep_layers = [[g_tracked_std[i],g_tracked_mean[i]] for i in match_indicies]
        self.gen_deep_layers = [x for y in self.gen_deep_layers for x in y]
        self.disc_deep_layers = [d_tracked[i] for i in match_indicies]
        
        g_final = self.G.functional_model
        d_final = self.D.functional_model
        self.generator = Model(inputs=self.G.input,outputs=[g_final,*self.gen_deep_layers])
        self.discriminator = Model(inputs=self.D.input,outputs=[d_final,*self.disc_deep_layers])

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
        return [mean_error,std_error]
        
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
            
            labels = tf.concat([self.real_label,self.fake_label],axis=0)
            disc_results = tf.concat([disc_real_results,disc_gen_results],axis=0)
            for metric in self.D.metrics:
                if metric.name == "mean":
                    metric.update_state(disc_results)
                else:
                    metric.update_state(labels,disc_results)
                out.append(metric.result())
            
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.D.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return out
