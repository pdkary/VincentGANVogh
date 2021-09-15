from trainers.AbstractTrainer import AbstractTrainer
import tensorflow as tf


class GenTapeTrainer(AbstractTrainer):
    def train_generator(self,source_input, gen_input):
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(gen_input, training=False)
            fake_out = self.discriminator(generated_images, training=True)
            
            g_loss = self.G.loss_function(self.gen_label, fake_out)
            out = [g_loss]
            
            for metric in self.gen_metrics:
                metric.update_state(self.gen_label,fake_out)
                out.append(metric.result())
            
            gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.G.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return out

    def train_discriminator(self, disc_input, gen_input):
        with tf.GradientTape() as disc_tape:
            generated_images = self.generator(gen_input, training=False)
            real_out = self.discriminator(disc_input, training=True)
            fake_out = self.discriminator(generated_images, training=True)
            
            labels = tf.concat([self.real_label,self.fake_label],axis=0)
            output = tf.concat([real_out,fake_out],axis=0)

            d_loss = self.D.loss_function(labels,output)
            out = [d_loss]
            
            for metric in self.disc_metrics:
                metric.update_state(labels,output)
                out.append(metric.result())
            
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.D.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return out
