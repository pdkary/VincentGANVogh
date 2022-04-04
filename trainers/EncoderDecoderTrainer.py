import tensorflow as tf
import tensorflow.keras.backend as K
from config.TrainingConfig import GanOutput, GanTrainingConfig, GanTrainingResult
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
            #get initial generated image
            noise_to_img: GanOutput = self.get_gen_output(gen_input,training=True)
            # get encoded gen and encoded real
            encoded_gen: GanOutput = self.get_disc_output(noise_to_img.result,training=True)
            encoded_real: GanOutput = self.get_disc_output(source_input,training=True)

            #adain normalize the encoded generated image
            enc_real_mean, enc_real_std = get_mean_std(encoded_real.result)
            enc_gen_mean, enc_gen_std = get_mean_std(encoded_gen.result)
            adain_enc_gen = enc_real_std*(encoded_gen.result - enc_gen_mean)/enc_gen_std + enc_real_mean
            
            #decode normalized encoding
            adain_dec_gen: GanOutput = self.get_gen_output(adain_enc_gen,training=True)

            #re-encode the new decoded image
            adain_reenc_gen: GanOutput = self.get_disc_output(adain_dec_gen.result,training=True)
            adain_reenc_mean, adain_reenc_std = get_mean_std(adain_reenc_gen.result)
            #content loss is diff between new encoding and real encoding
            g_loss = self.gen_loss_function(encoded_real.result,adain_reenc_gen.result)
            #style loss is diff of means and stds between
            g_loss += self.style_loss_coeff*self.gen_loss_function(enc_real_mean,adain_reenc_mean)
            g_loss += self.style_loss_coeff*self.gen_loss_function(enc_real_std,adain_reenc_std)
            #do gradients
            gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        
        metrics = []
        for metric in self.g_metrics:
            metric.update_state(adain_reenc_gen.result)
            metrics.append(metric.result())
        return GanTrainingResult(g_loss,metrics)


    def train_discriminator(self, disc_input, gen_input):
        with tf.GradientTape() as disc_tape:
            #get gen image
            noise_to_img: GanOutput = self.get_gen_output(gen_input,training=True)
            #encode both
            encoded_gen: GanOutput = self.get_disc_output(noise_to_img.result,training=True)
            encoded_real: GanOutput = self.get_disc_output(disc_input,training=True)
            #ones right, ones wrong, pretty simple
            d_loss = self.disc_loss_function(tf.ones_like(encoded_gen.result)*self.fake_label,encoded_gen.result)
            d_loss += self.disc_loss_function(tf.ones_like(encoded_real.result)*self.real_label,encoded_real.result)
            #do gradients    
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        metrics = []
        for metric in self.d_metrics:
            metric.update_state(encoded_real.result)
            metrics.append(metric.result())
        return GanTrainingResult(d_loss,metrics)

