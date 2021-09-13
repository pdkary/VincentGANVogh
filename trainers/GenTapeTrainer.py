from trainers.AbstractTrainer import AbstractTrainer
import numpy as np
import tensorflow as tf


class GenTapeTrainer(AbstractTrainer):
    def train_generator(self, gen_input):
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(gen_input, training=False)
            fake_out = self.discriminator(generated_images, training=True)
            g_loss = self.G.loss_function(self.gen_label, fake_out)
            self.gen_accuracy.update_state(self.gen_label,fake_out)
            g_avg = self.gen_accuracy.result()
            gradients_of_generator = gen_tape.gradient(
                g_loss, self.generator.trainable_variables)
            self.G.gen_optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables))
        return g_loss, g_avg

    def train_discriminator(self, disc_input, gen_input):
        with tf.GradientTape() as disc_tape:
            generated_images = self.generator(gen_input, training=False)
            real_out = self.discriminator(disc_input, training=True)
            fake_out = self.discriminator(generated_images, training=True)

            real_loss = self.D.loss_function(self.real_label, real_out)
            fake_loss = self.D.loss_function(self.fake_label, fake_out)
            d_loss = (real_loss + fake_loss)/2
            self.disc_accuracy.update_state(self.real_label,real_out)
            self.disc_accuracy.update_state(self.fake_label,fake_out)
            d_avg = self.disc_accuracy()

            gradients_of_discriminator = disc_tape.gradient(
                d_loss, self.discriminator.trainable_variables)
            self.D.disc_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return d_loss, d_avg
