from trainers.AbstractTrainer import AbstractTrainer
import tensorflow as tf


class CombinedTrainer(AbstractTrainer):
    def train_generator(self, gen_input):
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(gen_input, training=False)
            fake_out = self.discriminator(generated_images, training=True)
            g_loss = self.G.loss_function(self.gen_label, fake_out)
            self.gen_accuracy.update_state(self.gen_label,fake_out)
            g_avg = self.gen_accuracy.result()
            gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.G.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return g_loss, g_avg

    def train_discriminator(self, disc_input, gen_input):
        generated_images = self.generator(gen_input, training=False)
        real_loss, real_avg = self.discriminator.train_on_batch(disc_input, self.real_label)
        fake_loss, fake_avg = self.discriminator.train_on_batch(generated_images, self.fake_label)
        loss = (real_loss + fake_loss)/2
        avg = (real_avg + fake_avg)/2
        return loss, avg
