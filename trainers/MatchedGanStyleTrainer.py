from typing import List
import keras.backend as K
import tensorflow as tf
from config.TrainingConfig import GanTrainingConfig
from layers.AdaptiveInstanceNormalization import adain
from layers.GanInput import RealImageInput
from models.Discriminator import Discriminator
from models.Generator import Generator

from trainers.AbstractTrainer import AbstractTrainer
from tensorflow.keras.models import Model

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
        self.generator = Model(inputs=self.G.input,outputs=[g_final,*self.g_features])
        self.discriminator = Model(inputs=self.D.input,outputs=[d_final,*self.d_features])
        
    def get_style_loss(self,output_style,desired_style):
        out_2_desired = list(zip(output_style,desired_style))
        ada_outs = [adain(o,d) for o,d in out_2_desired]
        gen_2_ada = list(zip(output_style,ada_outs))
        return [self.G.loss_function(g,a) for g,a in gen_2_ada]
        
    def train_generator(self,source_input, gen_input):
        with tf.GradientTape() as gen_tape:
            gen_out = self.generator(gen_input,training=True)
            gen_images,gen_style = gen_out[0],gen_out[1:]
            
            disc_out = self.discriminator(gen_images, training=False)
            disc_content,disc_style = disc_out[0],disc_out[1:]
            
            content_loss = self.G.loss_function(self.gen_label, disc_content)
            style_losses = self.get_style_loss(gen_style,reversed(disc_style))
            
            g_loss = [content_loss,*style_losses]
            out = [content_loss]
            
            for metric in self.gen_metrics:
                metric.update_state(self.gen_label,disc_out)
                out.append(metric.result())
            
            gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.G.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return out

    def train_discriminator(self, disc_input, gen_input):
        with tf.GradientTape() as disc_tape:
            gen_out = self.generator(gen_input,training=True)
            gen_images,gen_style = gen_out[0],gen_out[1:]
            
            print("gen_images shape: ",gen_images.shape)
            print("gen_style shape: ",[x.shape for x in gen_style])
            
            disc_real = self.discriminator(disc_input, training=True)
            disc_gen_out = self.discriminator(gen_images, training=True)
            disc_real_content, disc_real_style = disc_real[0],disc_real[1:]
            disc_gen_content, disc_gen_style = disc_gen_out[0],disc_gen_out[1:]
            
            real_content_loss = self.D.loss_function(self.disc_labels[0], disc_real_content)
            fake_content_loss = self.D.loss_function(self.disc_labels[1], disc_gen_content)
            
            style_losses = self.get_style_loss(disc_gen_style,disc_real_style)
            content_loss = (real_content_loss + fake_content_loss)/2
            out = [content_loss,*style_losses]
            
            for metric in self.disc_metrics:
                metric.update_state(self.real_label,disc_real)
                metric.update_state(self.fake_label,disc_gen_out)
                out.append(metric.result())
            
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.D.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return out
