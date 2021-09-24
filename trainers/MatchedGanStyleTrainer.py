from typing import List
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.losses import Loss
from config.TrainingConfig import GanTrainingConfig
from layers.GanInput import RealImageInput
from models.Discriminator import Discriminator
from models.Generator import Generator

from trainers.AbstractTrainer import AbstractTrainer
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class MatchedGanStyleTrainer(AbstractTrainer):
    def __init__(self, 
                 generator: Generator,
                 discriminator: Discriminator,
                 gan_training_config: GanTrainingConfig,
                 style_loss_function: Loss,
                 image_sources: List[RealImageInput]):
        super().__init__(generator, discriminator, gan_training_config, image_sources)
        self.style_loss_function = style_loss_function
        gen_act = self.G.conv_activation
        disc_act = self.D.conv_activation
        self.matched_layers = set(gen_act.layer_dict.keys()) & set(disc_act.layer_dict.keys())
        print("MATCHING LAYERS: \n",self.matched_layers)
        
        self.disc_deep_layers = [disc_act.layer_dict[x] for x in self.matched_layers]        
        self.gen_deep_layers = [gen_act.layer_dict[x] for x in self.matched_layers]
        ##flatten 
        self.disc_deep_layers = [x.output for y in self.disc_deep_layers for x in y]        
        self.gen_deep_layers = [x.output for y in self.gen_deep_layers for x in y]
        
        self.null_style_loss = tf.constant([0.0 for i in self.disc_deep_layers],dtype=tf.float32)
        
        g_final = self.G.functional_model
        d_final = self.D.functional_model
        self.generator = Model(inputs=self.G.input,outputs=[g_final,*self.gen_deep_layers])
        self.discriminator = Model(inputs=self.D.input,outputs=[d_final,*self.disc_deep_layers])
        self.G.metric_labels = ["G_Style_loss"] + self.G.metric_labels
        self.plot_labels = ["G_Loss","D_Loss",*self.G.metric_labels,*self.D.metric_labels]

    
    def save(self,epoch):
        preview_seed = self.G.get_validation_input(self.preview_size)
        generated_images = np.array(self.generator.predict(preview_seed)[0])
        self.image_sources[0].save(epoch, generated_images, self.preview_rows, self.preview_cols, self.preview_margin)
        
    def get_deep_style_loss(self,content_src,style_src):
        src_2_dest = list(zip(content_src,style_src))
        return [self.get_style_loss(s,d) for s,d in src_2_dest]
    
    def get_style_loss(self,content_img,style_img,axis=[1,2,3]):
        mu_si = lambda x: (K.mean(x,axis),K.std(x,axis))
        mu_c, si_c = mu_si(content_img)
        mu_s, si_s = mu_si(style_img)
        mean_loss = self.style_loss_function(mu_s,mu_c)
        std_loss = self.style_loss_function(si_s,si_c)
        return mean_loss + std_loss
        
    def train_generator(self,source_input, gen_input):
        with tf.GradientTape() as gen_tape:
            gen_out = self.generator(gen_input,training=True)
            gen_images,gen_deep_layers = gen_out[0],gen_out[1:]
            
            disc_out = self.discriminator(gen_images, training=False)
            disc_results,disc_deep_layers = disc_out[0],disc_out[1:]
            
            content_loss = self.G.loss_function(self.gen_label, disc_results)
            deep_style_losses = self.get_deep_style_loss(gen_deep_layers,disc_deep_layers)
            
            total_loss = content_loss
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
            
            disc_real_out = self.discriminator(disc_input, training=True)[0]
            disc_gen_out = self.discriminator(gen_out, training=True)[0]
            
            real_content_loss = self.D.loss_function(self.real_label, disc_real_out)
            fake_content_loss = self.D.loss_function(self.fake_label, disc_gen_out)
            
            loss = (real_content_loss + fake_content_loss)/2
            d_loss = [loss, *self.null_style_loss]
            out = [loss]
            
            for metric in self.D.metrics:
                if metric.name == "mean":
                    metric.update_state(disc_real_out)
                    metric.update_state(disc_gen_out)
                else:
                    metric.update_state(self.real_label,disc_real_out)
                    metric.update_state(self.fake_label,disc_gen_out)
                out.append(metric.result())
            
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.D.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return out
