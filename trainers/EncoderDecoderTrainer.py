from typing import List

import tensorflow as tf
import tensorflow.keras.backend as K
from config.TrainingConfig import GanOutput, GanTrainingConfig
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

    def train(self, epochs, printerval):
        for epoch in range(epochs):
            gen_input = self.G.get_training_batch(self.batch_size)
            real_train_input = self.D.get_training_batch(self.batch_size)
            real_test_input = self.D.get_validation_batch(self.batch_size)

            with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
                #generate image from noise
                noise_to_img: GanOutput = self.get_gen_output(gen_input,training=True)
                
                #encode generated and real images
                encoded_gen: GanOutput = self.get_disc_output(noise_to_img.result,training=True)
                encoded_real: GanOutput = self.get_disc_output(real_train_input,training=True)
                encoded_test: GanOutput = self.get_disc_output(real_test_input,training=True)

                #adain the encoded generator
                enc_real_mean, enc_real_std = get_mean_std(encoded_real.result)
                enc_gen_mean, enc_gen_std = get_mean_std(encoded_gen.result)

                adain_enc_gen = enc_real_std*(encoded_gen.result - enc_gen_mean)/enc_gen_std + enc_real_mean
                adain_dec_gen: GanOutput = self.get_gen_output(adain_enc_gen,training=True)

                adain_reenc_gen: GanOutput = self.get_disc_output(adain_dec_gen.result,training=True)
                
                #get gen loss by diff of expected label and re-encoded adain output
                g_loss = self.gen_loss_function(tf.ones_like(adain_reenc_gen.result)*self.gen_label,adain_reenc_gen.result)

                #get disc loss by sum of real (test) images with labels and adain re-encoded adain gen output with labels
                d_loss = self.disc_loss_function(tf.ones_like(adain_reenc_gen.result)*self.fake_label,adain_reenc_gen.result)
                d_loss += self.disc_loss_function(tf.ones_like(encoded_test.result)*self.real_label,encoded_test.result)

                d_metrics, g_metrics = [], []
                for metric in self.g_metrics:
                    metric.update_state(adain_reenc_gen.result)
                    g_metrics.append(metric.result())

                for metric in self.d_metrics:
                    metric.update_state(encoded_real.result)
                    d_metrics.append(metric.result())
                
                if self.plot:
                    self.gan_plotter.batch_update([g_loss, d_loss, *g_metrics, *d_metrics])
                
                if epoch % printerval == 0:
                    name = "train-"+str(epoch)
                    data_helper: DataHelper = self.D.gan_input.data_helper
                    data_helper.save_images(name,adain_dec_gen.result,2,self.batch_size//2,self.preview_margin)
                    
                if epoch >= 10 and self.plot:
                    self.gan_plotter.log_epoch()
                
                gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
                self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
                gradients_of_discriminator = disc_tape.gradient(g_loss, self.discriminator.trainable_variables)
                self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

