from typing import List
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.losses import MSE
from config.TrainingConfig import GanTrainingConfig
from layers.AdaptiveInstanceNormalization import adain
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
                 image_sources: List[RealImageInput]):
        super().__init__(generator, discriminator, gan_training_config, image_sources)
        self.gen_act = self.G.gen_layers[0].activation
        self.disc_act = self.D.disc_conv_layers[0].activation
        g_features = [self.gen_act.find_by_size(x) for x in self.G.layer_sizes]
        self.g_features = [y.output for x in g_features for y in x]
        d_features = [self.disc_act.find_by_size(x) for x in self.D.layer_sizes]
        self.d_features = [y.output for x in d_features for y in x]
        
        g_final = self.G.functional_model
        d_final = self.D.functional_model
        self.null_style_loss = tf.constant([0.0 for i in self.d_features])
        self.generator = Model(inputs=self.G.input,outputs=[g_final,*self.g_features])
        self.discriminator = Model(inputs=self.D.input,outputs=[d_final,*self.d_features])
    
    def save(self,epoch):
        preview_seed = self.G.get_validation_input(self.preview_size)
        generated_images = np.array(self.generator.predict(preview_seed)[0])
        self.image_sources[0].save(epoch, generated_images, self.preview_rows, self.preview_cols, self.preview_margin)
        
    def get_deep_style_loss(self,content_src,style_src):
        src_2_dest = list(zip(content_src,style_src))
        return [self.get_style_loss(s,d) for s,d in src_2_dest]
    
    def get_style_loss(self,content_img,style_img,axis=[1,2]):
        ada_content = adain(content_img,style_img,axis)
        return tf.losses.mean_squared_error(content_img,ada_content)
        
    def train_generator(self,source_input, gen_input):
        with tf.GradientTape() as gen_tape:
            gen_out = self.generator(gen_input,training=True)
            gen_images,gen_deep_layers = gen_out[0],gen_out[1:]
            
            disc_gen_out = self.discriminator(gen_images, training=False)[0]
            disc_real = self.discriminator(source_input, training=False)
            disc_real_out,disc_real_deep_layers = disc_real[0], disc_real[1:]
            
            style_loss = self.get_style_loss(disc_real_out,disc_gen_out,axis=None])
            deep_style_losses = self.get_deep_style_loss(gen_deep_layers,reversed(disc_real_deep_layers))
            content_loss = self.G.loss_function(self.gen_label, disc_gen_out)
            g_loss = [content_loss + style_loss,*deep_style_losses]
            out = [g_loss[0]]
            
            for metric in self.gen_metrics:
                metric.update_state(self.gen_label,disc_gen_out)
                out.append(metric.result())
            
            gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.G.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return out

    def train_discriminator(self, disc_input, gen_input):
        with tf.GradientTape() as disc_tape:
            gen_out = self.generator(gen_input,training=False)[0]
            
            disc_real = self.discriminator(disc_input, training=True)[0]
            disc_gen = self.discriminator(gen_out, training=True)[0]
            
            real_content_loss = self.D.loss_function(self.real_label, disc_real)
            fake_content_loss = self.D.loss_function(self.fake_label, disc_gen)
            
            content_loss = (real_content_loss + fake_content_loss)/2
            d_loss = [content_loss,*self.null_style_loss]
            out = [d_loss[0]]
            
            for metric in self.disc_metrics:
                metric.update_state(self.real_label,disc_real)
                metric.update_state(self.fake_label,disc_gen)
                out.append(metric.result())
            
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.D.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return out
