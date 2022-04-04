from typing import List

import tensorflow as tf
import tensorflow.keras.backend as K
from config.TrainingConfig import GanOutput, GanTrainingConfig, GanTrainingResult
from helpers.DataHelper import DataHelper
from models.Discriminator import Discriminator
from models.Generator import Generator
from trainers.AbstractTrainer import AbstractTrainer

def get_mean_std(input_tensor,axis=0):
    return K.mean(input_tensor,axis=axis), K.std(input_tensor,axis=axis)

class EncoderDecoderTrainer(AbstractTrainer):
    def __init__(self, 
                 generator: Generator,
                 discriminator: Discriminator,
                 gan_training_config: GanTrainingConfig):
        super().__init__(generator, discriminator, gan_training_config)
        print(generator.input_shape)
        print(discriminator.output_shape)
        assert(generator.input_shape == discriminator.output_shape)
    
    def train_generator(self, source_input, gen_input):
        with tf.GradientTape() as gen_tape:
            noise_to_img: GanOutput = self.get_gen_output(gen_input,training=True)
            encoded_gen: GanOutput = self.get_disc_output(noise_to_img.result,training=True)
            encoded_real: GanOutput = self.get_disc_output(source_input,training=True)

            enc_real_mean, enc_real_std = get_mean_std(encoded_real.result)
            enc_gen_mean, enc_gen_std = get_mean_std(encoded_gen.result)

            adain_enc_gen = enc_real_std*(encoded_gen.result - enc_gen_mean)/enc_gen_std + enc_real_mean
            adain_dec_gen: GanOutput = self.get_gen_output(adain_enc_gen,training=True)
            adain_reenc_gen: GanOutput = self.get_disc_output(adain_dec_gen.result,training=True)
            
            g_loss = self.gen_loss_function(tf.ones_like(adain_reenc_gen.result)*self.gen_label,adain_reenc_gen.result)
            gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
        metrics = []
        for metric in self.d_metrics:
            metric.update_state(adain_reenc_gen.result)
            metrics.append(metric.result())
        return GanTrainingResult(g_loss,metrics)


    def train_discriminator(self, disc_input, gen_input):
        with tf.GradientTape() as disc_tape:
            noise_to_img: GanOutput = self.get_gen_output(gen_input,training=True)
            
            encoded_gen: GanOutput = self.get_disc_output(noise_to_img.result,training=True)
            encoded_real: GanOutput = self.get_disc_output(disc_input,training=True)

            d_loss = self.disc_loss_function(tf.ones_like(encoded_gen.result)*self.fake_label,encoded_gen.result)
            d_loss += self.disc_loss_function(tf.ones_like(encoded_real.result)*self.real_label,encoded_real.result)
                
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        metrics = []
        for metric in self.d_metrics:
            metric.update_state(encoded_real.result)
            metrics.append(metric.result())
        return GanTrainingResult(d_loss,metrics)

