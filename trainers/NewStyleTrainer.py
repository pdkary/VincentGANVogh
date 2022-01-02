import tensorflow as tf
import keras.backend as K
from config.TrainingConfig import GanTrainingResult
from trainers.AbstractTrainer import AbstractTrainer

class NewStyleTrainer(AbstractTrainer):

    def train_generator(self, source_input, gen_input):
        with tf.GradientTape() as gen_tape:
            gen_output = self.generator(gen_input,training=True)
            
            if self.G.tracked_layers == {}:
                gen_images = gen_output
                gen_stds = []
                gen_means = []
            else:
                gen_images = gen_output[0]
                gen_tracked_layers = gen_output[1:]
                gen_stds =  [K.std(x,self.G.std_dims) for x in gen_tracked_layers]
                gen_means = [K.mean(x,self.G.std_dims) for x in gen_tracked_layers]

            disc_out = self.discriminator(gen_images, training=False)

            if self.D.tracked_layers == {}:
                disc_results = disc_out
            else:
                disc_results = disc_out[0]
                
            content_loss = self.gen_loss_function(self.gen_label, disc_results)
            content_loss += self.gen_loss_function(source_input,gen_images)

            g_loss = content_loss
            metrics = []
            
            for metric in self.g_metrics:
                if metric.name == "mean":
                    metric.update_state(disc_results)
                else:
                    metric.update_state(self.gen_label,disc_results)
                metrics.append(metric.result())
            
            gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return GanTrainingResult(g_loss,metrics,gen_stds,gen_means)

    def train_discriminator(self, source_input, gen_input):
        with tf.GradientTape() as gen_tape:
            gen_output = self.generator(gen_input,training=True)
            
            if self.G.tracked_layers == {}:
                gen_images = gen_output
            else:
                gen_images = gen_output[0]

            disc_out = self.discriminator(gen_images, training=False)

            if self.D.tracked_layers == {}:
                disc_results = disc_out
                disc_stds = []
                disc_means = []
            else:
                disc_results = disc_out[0]
                disc_tracked_layers = disc_out[1:]
                disc_stds =  [K.std(x,self.D.std_dims) for x in disc_tracked_layers]
                disc_means = [K.mean(x,self.D.std_dims) for x in disc_tracked_layers]
                
            content_loss = self.gen_loss_function(self.gen_label, disc_results)
            content_loss += self.gen_loss_function(source_input,gen_images)

            d_loss = content_loss
            metrics = []
            
            for metric in self.d_metrics:
                if metric.name == "mean":
                    metric.update_state(disc_results)
                else:
                    metric.update_state(self.gen_label,disc_results)
                metrics.append(metric.result())
            
            gradients_of_generator = gen_tape.gradient(d_loss, self.generator.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return GanTrainingResult(d_loss,metrics,disc_stds,disc_means)