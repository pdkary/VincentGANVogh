from abc import ABC, abstractmethod
from typing import List

import numpy as np
from config.TrainingConfig import GanTrainingConfig, GanTrainingResult
from helpers.DataHelper import DataHelper
from models.Discriminator import Discriminator
from models.Generator import Generator
from tensorflow.keras.models import Model

def flatten(arr: List):
    return [x for y in arr for x in y]

class AbstractTrainer(GanTrainingConfig, ABC):
    def __init__(self,
                 generator:    Generator,
                 discriminator:   Discriminator,
                 gan_training_config: GanTrainingConfig):
        GanTrainingConfig.__init__(self, **gan_training_config.__dict__)
        self.G: Generator = generator
        self.D: Discriminator = discriminator

        self.preview_size = self.preview_cols*self.preview_rows
        label_shape = (self.batch_size, self.D.dense_layers[-1].size)
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
    
    def compile(self):
        GI,GO = self.G.input,self.G.build()
        DI,DO = self.D.input,self.D.build()

        G_STDS = [x.std for x in self.G.tracked_layers.values()]
        G_MEANS = [x.mean for x in self.G.tracked_layers.values()]
        D_STDS = [x.std for x in self.D.tracked_layers.values()]
        D_MEANS = [x.mean for x in self.D.tracked_layers.values()]

        g_outs = [GO,*G_STDS,*G_MEANS]
        d_outs = [DO,*D_STDS,*D_MEANS]
        
        self.generator = Model(inputs=GI,outputs=g_outs,name="Generator")
        self.generator.compile(optimizer=self.gen_optimizer,
                               loss=self.gen_loss_function,
                               metrics=self.g_metrics)
        self.generator.summary()
        
        self.discriminator = Model(inputs=DI,outputs=d_outs,name="Discriminator")
        self.discriminator.compile(optimizer=self.disc_optimizer,
                                   loss=self.disc_loss_function,
                                   metrics=self.d_metrics)
        self.discriminator.summary()
    
    @abstractmethod
    def train_generator(self, source_input, gen_input) -> GanTrainingResult:
        pass

    @abstractmethod
    def train_discriminator(self, source_input, gen_input) -> GanTrainingResult:
        pass

    def save_images(self,name):
        data_helper: DataHelper = self.D.gan_input.data_helper
        gen_input = self.G.get_validation_batch(self.preview_size)
        gen_images = self.generator.predict(gen_input)
        data_helper.save_images(name,gen_images,self.preview_rows,self.preview_cols,self.preview_margin)

    def save_generator(self, epoch):
        filename = self.model_name + str(epoch)
        self.generator.save(self.model_output_path + filename)

    def train(self, epochs, printerval):
        for epoch in range(epochs):
            if self.plot:
                self.gan_plotter.start_epoch()
            
            train_input = self.D.gan_input.get_training_batch(self.batch_size)
            test_input = self.D.gan_input.get_validation_batch(self.batch_size)
            gen_input = self.G.get_training_batch(self.batch_size)
            
            DO = self.train_discriminator(test_input, gen_input)
            GO = self.train_generator(train_input, gen_input)

            if self.plot:
              self.gan_plotter.batch_update([GO.loss, DO.loss, *GO.metrics, *DO.metrics])
            
            if epoch % printerval == 0:
                self.save_images("train-"+str(epoch))
                
            if epoch >= 10 and self.plot:
                self.gan_plotter.log_epoch()

    def train_n_eras(self, eras, epochs, printerval, ma_size):
        if self.plot:
            from helpers.GanPlotter import GanPlotter
            self.gan_plotter = GanPlotter(moving_average_size=ma_size, labels=self.plot_labels)
        for i in range(eras):
            self.train(epochs, printerval)
            self.save_generator((i+1)*epochs)
