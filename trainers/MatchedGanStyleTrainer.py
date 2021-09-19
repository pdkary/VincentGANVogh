from layers.AdaptiveInstanceNormalization import adain
import keras.backend as K
from trainers.AbstractTrainer import AbstractTrainer
import tensorflow as tf

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
        self.g_features = [y for x in g_features for y in x]
        d_features = [self.disc_act.find_by_size(x) for x in self.D.layer_sizes]
        self.d_features = [y for x in d_features for y in x]
        
        g_final = self.G.functional_model
        d_final = self.D.functional_model
        self.generator = Model(inputs=self.G.input,outputs=[g_final,*self.g_features])
        self.discriminator = Model(inputs=self.D.input,outputs=[d_final,*self.d_features])
        
    # def get_style_loss(self,content_loss_arr):
    #     loss = 0.0
    #     for x in self.G.layer_sizes:
    #         gen_activations = self.G.gen_layers[0].activation.find_by_size(x)
    #         disc_activations = self.D.disc_conv_layers[0].activation.find_by_size(x)
    #         assert len(gen_activations) == len(disc_activations)
    #         if len(gen_activations) > 0:
    #             gen_outs = [ga.output for ga in gen_activations]
    #             print("gen_out_shapes: ",[x.shape for x in gen_outs])
    #             disc_outs = [da.output for da in disc_activations]
    #             print("disc_out_shapes: ",[x.shape for x in disc_outs])
    #             gen_2_disc = list(zip(gen_outs,disc_outs))
    #             ada_outs = [adain(g,d) for g,d in gen_2_disc]
    #             print("ada_out_shapes: ",[x.shape for x in ada_outs])
    #             gen_2_ada = list(zip(gen_outs,ada_outs))
    #             layer_loss = [self.G.loss_function(g,a) for g,a in gen_2_ada]
    #             print("layer_loss_shapes: ",[x.shape for x in layer_loss])
    #             loss += K.sum(layer_loss)
    #     out = loss*tf.ones_like(content_loss_arr)
    #     print("style loss: ",out)
    #     return out
    
    def train_generator(self,source_input, gen_input):
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(gen_input, training=False)
            fake_out = self.discriminator(generated_images, training=True)
            
            content_loss = self.G.loss_function(self.gen_label, fake_out)
            # style_loss = self.get_style_loss(content_loss)
            g_loss = content_loss + style_loss
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
            print(generated_images.shape)
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