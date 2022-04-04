from abc import ABC, abstractmethod
from typing import List

import numpy as np
from config.TrainingConfig import GanOutput, GanTrainingConfig, GanTrainingResult
from helpers.DataHelper import DataHelper
from models.Discriminator import Discriminator
from models.Generator import Generator
from tensorflow.keras.models import Model

def flatten(arr: List):
    return [x for y in arr for x in y]

def sort_gan_output(arr: List,feature_length:int):
    return [arr[0],arr[1:feature_length+1],arr[feature_length+1:]]

class AbstractTrainer(GanTrainingConfig, ABC):
    def __init__(self,
                 generator:    Generator,
                 discriminator:   Discriminator,
                 gan_training_config: GanTrainingConfig):
        GanTrainingConfig.__init__(self, **gan_training_config.__dict__)
        self.G: Generator = generator
        self.D: Discriminator = discriminator

        self.preview_size = self.preview_cols*self.preview_rows
        
        self.preview_args = {
            "preview_rows":self.preview_rows,
            "preview_cols":self.preview_cols,
            "preview_margin":self.preview_margin
        }
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
        
        g_outs = [GO,*self.G.feature_layers,*self.G.view_layers]
        d_outs = [DO,*self.D.feature_layers,*self.D.view_layers]
        
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
    def train(self, epochs, printerval):
        pass

    def get_gen_output(self,gen_input,training=True) -> GanOutput:
        output = self.generator(gen_input,training=True) if training else self.generator.predict(gen_input)
        if not self.G.feature_layers and not self.G.view_layers:
            return GanOutput(output)
        else:
            fl = len(self.G.feature_layers) # featuer length
            sorted_output = sort_gan_output(output,fl)
            return GanOutput(*sorted_output)

    def get_disc_output(self,disc_input,training=True) -> GanOutput:
        output = self.discriminator(disc_input,training=True) if training else self.discriminator.predict(disc_input)
        if not self.D.feature_layers and not self.D.view_layers:
            return GanOutput(output)
        else:
            fl = len(self.D.feature_layers)
            sorted_output = sort_gan_output(output,fl)
            return GanOutput(*sorted_output)

    def save_images(self,name):
        data_helper: DataHelper = self.D.gan_input.data_helper
        gen_input = self.G.get_validation_batch(self.preview_size)

        gen_output:GanOutput = self.get_gen_output(gen_input,training=False)

        for i,im in enumerate(gen_output.views):
            view_name = "sublayers/view_" + str(i) + "_" + name
            data_helper.save_images(view_name,im,**self.preview_args)

        data_helper.save_images(name,gen_output.result,**self.preview_args)

    def save_generator(self, epoch):
        filename = self.model_name + str(epoch)
        self.generator.save(self.model_output_path + filename)

    def train_n_eras(self, eras, epochs, printerval, ma_size):
        if self.plot:
            from helpers.GanPlotter import GanPlotter
            self.gan_plotter = GanPlotter(moving_average_size=ma_size, labels=self.plot_labels)
        for i in range(eras):
            self.train(epochs, printerval)
            self.save_generator((i+1)*epochs)
