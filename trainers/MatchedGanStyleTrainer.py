from layers.AdaptiveInstanceNormalization import adain
import keras.backend as K
from trainers.AbstractTrainer import AbstractTrainer
import tensorflow as tf

class MatchedGanStyleTrainer(AbstractTrainer):
    def get_style_loss(self,content_loss_arr):
        loss = 0.0
        for x in self.G.layer_sizes:
            gen_activations = self.G.gen_layers[0].activation.find_by_size(x)
            disc_activations = self.D.disc_conv_layers[0].activation.find_by_size(x)
            
            gen_outs = [ga.output for ga in gen_activations]
            ada_outs = [adain(gen_outs[i],disc_activations[i]) for i in disc_activations]
            loss += self.G.loss_function(gen_outs,ada_outs)
        return loss*tf.ones_like(content_loss_arr)
    
    def train_generator(self,source_input, gen_input):
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(gen_input, training=False)
            fake_out = self.discriminator(generated_images, training=True)
            
            content_loss = self.G.loss_function(self.gen_label, fake_out)
            style_loss = self.get_style_loss(content_loss)
            print(type(content_loss))
            print(type(style_loss))
            g_loss = content_loss + style_loss
            print(type(g_loss))
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

            real_loss = self.D.loss_function(self.real_label,real_out)
            fake_loss = self.D.loss_function(self.fake_label,fake_out)
            d_loss = (real_loss + fake_loss)/2
            out = [d_loss]
            
            for metric in self.disc_metrics:
                metric.update_state(self.real_label,real_out)
                metric.update_state(self.fake_label,fake_out)
                out.append(metric.result())
            
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.D.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return out