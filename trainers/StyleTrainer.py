from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K
from config.TrainingConfig import GanTrainingConfig
from models.Discriminator import Discriminator
from models.Generator import Generator
from tensorflow.keras.models import Model
from trainers.ViewableTrainer import ViewableTrainer

def flatten(arr: List):
    return [x for y in arr for x in y]

class StyleTrainer(ViewableTrainer):
    def __init__(self, 
                 generator: Generator,
                 discriminator: Discriminator,
                 gan_training_config: GanTrainingConfig):
        super().__init__(generator, discriminator, gan_training_config)
    
    def get_all_style_loss(self,content_std_arr,content_mean_arr,style_std_arr,style_mean_arr):
        src_2_dest = list(zip(content_std_arr,content_mean_arr,style_std_arr,style_mean_arr))
        return [self.get_style_loss(cs,cm,ss,sm) for cs,cm,ss,sm in src_2_dest]
    
    def get_style_loss(self,content_std,content_mean,s_std,s_mean):
        std_error = self.style_loss_coeff*self.style_loss_function(s_std,content_std)
        mean_error = self.style_loss_coeff*self.style_loss_function(s_mean,content_mean)
        return [std_error,mean_error]

    def compile(self):
        GI,GO = self.G.input,self.G.build()
        DI,DO = self.D.input,self.D.build()
        self.matched_keys = [g for g in self.G.tracked_layers.keys() if g in self.D.tracked_layers]
        self.gen_style_layers,self.disc_style_layers = [], []
        self.gen_view_layers,self.disc_view_layers = [],[]

        for key in self.matched_keys:
            G_data = self.G.tracked_layers[key]
            D_data = self.D.tracked_layers[key]
            self.gen_style_layers.extend( G_data["std_mean"])
            self.disc_style_layers.extend(D_data["std_mean"])
            if self.G.view_layers:
                self.gen_view_layers.append(G_data["view"])
            if self.D.view_layers:
                self.disc_view_layers.append(D_data["view"])
        
        print("TRACKING LAYERS")
        print(" gen: ",[x.shape for x in self.gen_style_layers])
        print("disc: ",[x.shape for x in self.disc_style_layers])
        print("VIEWING LAYERS")
        print(" gen: ",[x.shape for x in self.gen_view_layers])
        print("disc: ",[x.shape for x in self.disc_view_layers])

        self.gview_index = 1 + len(self.gen_style_layers)
        self.dview_index = 1 + len(self.disc_style_layers)

        self.generator = Model(inputs=GI,outputs=[GO,*self.gen_style_layers,*self.gen_view_layers],name="Generator")
        self.discriminator = Model(inputs=DI,outputs=[DO,*self.disc_style_layers,*self.disc_view_layers],name="Discriminator")

        self.generator.compile(optimizer=self.gen_optimizer,
                               loss=self.gen_loss_function,
                               metrics=self.g_metrics)
        self.discriminator.compile(optimizer=self.disc_optimizer,
                                   loss=self.disc_loss_function,
                                   metrics=self.d_metrics)
        self.generator.summary()
        self.discriminator.summary()            
        self.g_metric_labels = ["G_Style_loss"] + self.g_metric_labels
        self.plot_labels = ["G_Loss","D_Loss",*self.g_metric_labels,*self.d_metric_labels]

    def save_images(self,name):
        if self.D.view_layers or self.G.view_layers:
            self.save_views(name)
        elif not self.G.view_layers:
            preview_seed = self.G.get_validation_batch(self.preview_size)
            gen_images = self.generator.predict(preview_seed)[0]
            self.D.gan_input.save(name, gen_images, self.preview_rows, self.preview_cols, self.preview_margin)
        
    def get_all_style_loss(self,content_std_arr,content_mean_arr,style_std_arr,style_mean_arr):
        src_2_dest = list(zip(content_std_arr,content_mean_arr,style_std_arr,style_mean_arr))
        return [self.get_style_loss(cs,cm,ss,sm) for cs,cm,ss,sm in src_2_dest]
    
    def get_style_loss(self,content_std,content_mean,s_std,s_mean):
        std_error = self.style_loss_coeff*self.style_loss_function(s_std,content_std)
        mean_error = self.style_loss_coeff*self.style_loss_function(s_mean,content_mean)
        return [std_error,mean_error]

    def train_generator(self,source_input, gen_input):
        vi = self.gview_index
        with tf.GradientTape() as gen_tape:
            gen_out = self.generator(gen_input,training=True)
            gen_images, gen_style, gen_view = gen_out[0],gen_out[1:vi],gen_out[vi:]
            gen_style_std,gen_style_mean = gen_style[0::2],gen_style[1::2]
            
            disc_out = self.discriminator(gen_images, training=False)
            disc_results,disc_style,disc_view = disc_out[0],disc_out[1:vi],disc_out[vi:]
            disc_style_std,disc_style_mean = disc_style[0::2],disc_style[1::2]

            content_loss = self.gen_loss_function(self.gen_label, disc_results)

            source_std  = K.std(source_input,self.G.std_dims,keepdims=True)
            source_mean = K.mean(source_input,self.G.std_dims,keepdims=True)
            gen_std  = K.std(gen_images,self.G.std_dims,keepdims=True)
            gen_mean = K.mean(gen_images,self.G.std_dims,keepdims=True)
            style_loss = self.style_loss_coeff*((source_std - gen_std)**2 + (source_mean - gen_mean)**2)/2

            style_losses = self.get_all_style_loss(gen_style_std,gen_style_mean,disc_style_std,disc_style_mean) if len(self.matched_keys) > 0 else []
            view_losses = [tf.zeros_like(x) for x in gen_view]

            g_loss = [content_loss + style_loss,*style_losses,*view_losses]
            out = [content_loss, style_loss + np.sum(style_losses)]
            
            for metric in self.g_metrics:
                if metric.name == "mean":
                    metric.update_state(disc_results)
                else:
                    metric.update_state(self.gen_label,disc_results)
                out.append(metric.result())
            
            gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return out

    def train_discriminator(self, disc_input, gen_input):
        vi = self.gview_index
        with tf.GradientTape() as disc_tape:
            gen_out = self.generator(gen_input,training=False)
            gen_images, gen_style, gen_view = gen_out[0], gen_out[1:vi], gen_out[vi:]

            disc_gen_out = self.discriminator(gen_images, training=True)
            disc_gen_result,disc_gen_style,disc_gen_view = disc_gen_out[0],disc_gen_out[1:vi],disc_gen_out[vi:]

            disc_real_out = self.discriminator(disc_input, training=True)
            disc_real_result,disc_real_style,disc_real_view = disc_real_out[0],disc_real_out[1:vi],disc_real_out[vi:]
            
            content_loss = self.disc_loss_function(self.fake_label, disc_gen_result) + self.disc_loss_function(self.real_label, disc_real_result)
            style_losses = [tf.zeros_like(x) for x in disc_real_style]
            view_losses = [tf.zeros_like(x) for x in disc_real_view] 

            d_loss = [content_loss,*style_losses,*view_losses]
            out = [content_loss]
            
            for metric in self.d_metrics:
                m :Metric = metric
                if m.name == "mean":
                    m.update_state(disc_real_result)
                else:
                    m.update_state(self.real_label,disc_real_result)
                out.append(m.result())
            
            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return out
