from abc import ABC, abstractmethod

import numpy as np
from config.TrainingConfig import GanTrainingConfig
from models.Discriminator import Discriminator
from models.Generator import Generator
from tensorflow.keras.models import Model

class AbstractTrainer(GanTrainingConfig, ABC):
    def __init__(self,
                 generator:    Generator,
                 discriminator:   Discriminator,
                 gan_training_config: GanTrainingConfig):
        GanTrainingConfig.__init__(self, **gan_training_config.__dict__)
        self.G: Generator = generator
        self.D: Discriminator = discriminator

        self.preview_size = self.preview_cols*self.preview_rows
        label_shape = (self.batch_size, self.D.dense_layers[-1])
        self.real_label = np.full(label_shape,self.disc_labels[0])
        self.fake_label = np.full(label_shape,self.disc_labels[1])
        self.gen_label = np.full(label_shape, self.gen_label)

        self.g_metrics = [m() for m in self.metrics]
        self.d_metrics = [m() for m in self.metrics]
        self.g_metric_labels = ["G_" + str(m.name) for m in self.g_metrics]
        self.d_metric_labels = ["D_" + str(m.name) for m in self.d_metrics]
        self.plot_labels = ["G_Loss","D_Loss",*self.g_metric_labels,*self.d_metric_labels]
        self.model_output_path = self.D.gan_input.data_path + "/models"
        self.model_name = self.D.gan_input.model_name
    
    @abstractmethod
    def compile(self):
        GI,GO = self.G.input,self.G.build()
        DI,DO = self.D.input,self.D.build()
        self.generator = Model(inputs=GI,outputs=GO)
        self.generator.compile(optimizer=self.gen_optimizer,
                               loss=self.gen_loss_function,
                               metrics=self.g_metrics)
        self.generator.summary()
        
        self.discriminator = Model(inputs=DI,outputs=DO)
        self.discriminator.compile(optimizer=self.disc_optimizer,
                                   loss=self.disc_loss_function,
                                   metrics=self.d_metrics)
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
        for epoch in range(epochs):
            if self.plot:
                self.gan_plotter.start_epoch()
            
            source_input = self.D.gan_input.get_training_batch(self.batch_size)
            gen_input = self.G.get_training_batch(self.batch_size)
            
            disc_results = self.train_discriminator(source_input, gen_input)
            gen_results = self.train_generator(source_input, gen_input)

            d_loss,d_metrics = disc_results[0],disc_results[1:]
            g_loss,g_metrics = gen_results[0],gen_results[1:]

            if self.plot:
              self.gan_plotter.batch_update([g_loss, d_loss, *g_metrics, *d_metrics])
            
            if epoch % printerval == 0:
                self.save(epoch)
                
            if epoch >= 10 and self.plot:
                self.gan_plotter.log_epoch()

    def train_n_eras(self, eras, epochs, printerval, ma_size):
        if self.plot:
            from helpers.GanPlotter import GanPlotter
            self.gan_plotter = GanPlotter(moving_average_size=ma_size, labels=self.plot_labels)
        for i in range(eras):
            self.train(epochs, printerval)
            self.save_generator((i+1)*epochs)
