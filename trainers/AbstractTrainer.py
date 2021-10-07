from abc import ABC, abstractmethod
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from config.TrainingConfig import GanTrainingConfig
from inputs.GanInput import RealImageInput
from models.Discriminator import Discriminator
from models.Generator import Generator


class AbstractTrainer(GanTrainingConfig, ABC):
    def __init__(self,
                 generator:    Generator,
                 discriminator:   Discriminator,
                 gan_training_config: GanTrainingConfig):
        GanTrainingConfig.__init__(self, **gan_training_config.__dict__)
        self.G: Generator = generator
        self.D: Discriminator = discriminator

        self.preview_size = self.preview_cols*self.preview_rows
        label_shape = (self.batch_size, self.D.DM.dense_layers[-1])
        self.real_label = np.full(label_shape,self.disc_labels[0])
        self.fake_label = np.full(label_shape,self.disc_labels[1])
        self.gen_label = np.full(label_shape, self.gen_label)

        self.g_metrics = [m() for m in self.metrics]
        self.d_metrics = [m() for m in self.metrics]
        self.g_metric_labels = ["G_" + str(m.name) for m in self.metrics]
        self.d_metric_labels = ["D_" + str(m.name) for m in self.metrics]
        self.plot_labels = ["G_Loss","D_Loss",*self.g_metric_labels,*self.d_metric_labels]
        self.model_output_path = self.D.CM.gan_input.data_path + "/models"
        self.model_name = self.D.gan_input.model_name

    def compile(self):
        self.generator = Model(inputs=self.G.CM.inputs,outputs=self.G.functional_model,name="Generator")
        self.generator.compile(optimizer=self.gen_optimizer,loss=self.gen_loss_function,metrics=self.g_metrics)
        self.generator.summary()
        
        self.discriminator = Model(inputs=self.D.CM.inputs,outputs=self.D.functional_model,name="Discriminator")
        self.discriminator.compile(optimizer=self.disc_optimizer,loss=self.disc_loss_function,metrics=self.d_metrics)
        self.discriminator.summary()
    
    @abstractmethod
    def train_generator(self, source_input, gen_input):
        return 2*[0.0] + len(self.g_metrics)*[0.0]

    @abstractmethod
    def train_discriminator(self, source_input, gen_input):
        return 2*[0.0] + len(self.d_metrics)*[0.0]

    def save_generator(self, epoch):
        filename = self.model_name + str(epoch)
        self.generator.save(self.model_output_path + filename)

    def train(self, epochs, printerval):
        for epoch in tf.range(epochs):
            if self.plot:
                self.gan_plotter.start_epoch()

            d_loss, d_metrics = 0.0, [0.0 for i in self.d_metric_labels]
            g_loss, g_metrics = 0.0, [0.0 for i in self.g_metric_labels]
            
            for i in tf.range(self.disc_batches_per_epoch):
                source_input = self.D.gan_input.get_validation_batch(self.batch_size)
                gen_input = self.G.get_validation_batch(self.batch_size)
                batch_out = self.train_discriminator(source_input, gen_input)
                batch_loss,batch_metrics = batch_out[0],batch_out[1:]
                d_loss += batch_loss
                for i in tf.range(len(self.d_metric_labels)):
                    d_metrics[i] += batch_metrics[i]
                
            for i in tf.range(self.gen_batches_per_epoch):
                source_input = self.D.gan_input.get_training_batch(self.batch_size)
                gen_input = self.G.get_training_batch(self.batch_size)
                batch_out = self.train_generator(source_input,gen_input)
                batch_loss,batch_metrics = batch_out[0],batch_out[1:]
                g_loss += batch_loss
                for i in tf.range(len(self.g_metric_labels)):
                    g_metrics[i] += batch_metrics[i]

            if self.plot:
                d_loss /= self.disc_batches_per_epoch
                g_loss /= self.gen_batches_per_epoch
                d_metrics = [d/self.disc_batches_per_epoch for d in d_metrics]
                g_metrics = [g/self.gen_batches_per_epoch for g in g_metrics]
                self.gan_plotter.batch_update([g_loss, d_loss, *g_metrics, *d_metrics])
            
            if epoch % printerval == 0:
                self.save(epoch)
                
            if epoch >= 10 and self.plot:
                self.gan_plotter.log_epoch()

    def save(self,epoch):
        preview_seed = self.G.get_validation_batch(self.preview_size)
        generated_images = np.array(self.generator.predict(preview_seed))
        self.D.gan_input.save(epoch, generated_images, self.preview_rows, self.preview_cols, self.preview_margin)
    
    def train_n_eras(self, eras, epochs, printerval, ma_size):
        if self.plot:
            from helpers.GanPlotter import GanPlotter
            self.gan_plotter = GanPlotter(moving_average_size=ma_size, labels=self.plot_labels)
        for i in range(eras):
            self.train(epochs, printerval)
            self.save_generator((i+1)*epochs)
