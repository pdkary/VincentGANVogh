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

class GradTapeStyleTrainer(AbstractTrainer):
    def __init__(self, 
                 generator: Generator,
                 discriminator: Discriminator,
                 gan_training_config: GanTrainingConfig):
        super().__init__(generator, discriminator, gan_training_config)

    def compile(self):
        GI,GO = self.G.input,self.G.build()
        DI,DO = self.D.input,self.D.build()
        
        self.matched_keys = [g for g in self.G.tracked_layers.keys() if g in self.D.tracked_layers]
        self.gen_deep_layers = flatten([self.G.tracked_layers[i] for i in self.matched_keys])
        print("gen_deep_layers:")
        print([x.name for x in self.gen_deep_layers])
        self.disc_deep_layers = flatten([self.D.tracked_layers[i] for i in self.matched_keys])
        print("disc_deep_layers:")
        print([x.name for x in self.disc_deep_layers])
        
        self.generator = Model(inputs=GI,outputs=[GO,*self.gen_deep_layers])
        self.discriminator = Model(inputs=DI,outputs=[DO,*self.disc_deep_layers])

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
        self.nil_disc_style_loss = tf.constant([0.0 for i in self.disc_deep_layers],dtype=tf.float32)
        self.g_metric_labels = ["G_Style_loss"] + self.g_metric_labels
        self.plot_labels = ["G_Loss","D_Loss",*self.g_metric_labels,*self.d_metric_labels]
    
    def save(self,epoch):
        preview_seed = self.G.get_validation_batch(self.preview_size)
        generated_images = np.array(self.generator.predict(preview_seed)[0])
        self.D.gan_input.save(epoch, generated_images, self.preview_rows, self.preview_cols, self.preview_margin)
        
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
            if len(self.matched_keys) > 0:
                gen_images,gen_deep_mean_layers,gen_deep_std_layers = gen_out[0],gen_out[1::2],gen_out[2::2]
            else: 
                gen_images,gen_deep_mean_layers,gen_deep_std_layers = gen_out,None,None

            disc_out = self.discriminator(gen_images, training=False)
            
            if len(self.matched_keys) > 0:
                disc_results,disc_deep_layers = disc_out[0],disc_out[1:]
            else:
                disc_results,disc_deep_layers = disc_out,None
            
            content_loss = self.gen_loss_function(self.gen_label, disc_results)
            deep_style_losses = self.get_deep_style_loss(gen_deep_std_layers,gen_deep_mean_layers,disc_deep_layers) if len(self.matched_keys) > 0 else [] 
            
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
        print("Training Discriminator")
        print("D shape: ",disc_input.shape)
        print("D mean: ", np.mean(disc_input))
        print("D std: ", np.std(disc_input))
        print("G shape: ",gen_input.shape)
        print("G mean: ", np.mean(gen_input))
        print("G std: ", np.std(gen_input))
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
