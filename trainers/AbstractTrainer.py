from abc import ABC, abstractmethod
from typing import List
from layers.GanInput import RealImageInput
from config.TrainingConfig import DataConfig, GanTrainingConfig
from config.DiscriminatorConfig import DiscriminatorModelConfig
from config.GeneratorConfig import GeneratorModelConfig
from models.Generator import Generator
from models.Discriminator import Discriminator
import numpy as np
import tensorflow as tf

class AbstractTrainer(GanTrainingConfig, ABC):
    def __init__(self,
                 gen_model_config:    GeneratorModelConfig,
                 disc_model_config:   DiscriminatorModelConfig,
                 gan_training_config: GanTrainingConfig,
                 data_configs:         List[DataConfig]):
        GanTrainingConfig.__init__(self, **gan_training_config.__dict__)
        self.G: Generator = Generator(gen_model_config)
        self.D: Discriminator = Discriminator(disc_model_config)
        self.image_sources: List[RealImageInput] = [RealImageInput(d) for d in data_configs]
        self.preview_size = self.preview_cols*self.preview_rows

        label_shape = (self.batch_size, self.D.output_dim)
        self.real_label = self.disc_labels[0]*tf.ones(shape=label_shape)
        self.fake_label = self.disc_labels[1]*tf.ones(shape=label_shape)
        self.gen_label = self.gen_label*tf.ones(shape=label_shape)

        self.generator = self.G.build()
        self.discriminator = self.D.build()
        self.model_output_path = data_configs[0].data_path + "/models"
        d_metric_labels = ["D_" + m.name for m in self.metrics]
        g_metric_labels = ["G_" + m.name for m in self.metrics]
        self.plot_labels = ["D_Loss","G_Loss",*d_metric_labels,*g_metric_labels]
        self.metrics = [m() for m in self.metrics]

        for source in self.image_sources:
            source.load()

    @abstractmethod
    def train_generator(self, gen_input):
        return 2*[0.0] + len(self.metrics)*[0.0]

    @abstractmethod
    def train_discriminator(self, source_input, gen_input):
        return 2*[0.0] + len(self.metrics)*[0.0]

    def save_generator(self, epoch):
        filename = self.image_sources[0].data_helper.model_name + str(epoch)
        self.generator.save(self.model_output_path + filename)

    def train(self, epochs, printerval):
        for epoch in range(epochs):
            if self.plot:
                self.gan_plotter.start_epoch()

            for source in self.image_sources:
                d_loss, d_metrics = 0.0, [0.0 for i in self.metrics]
                g_loss, g_metrics = 0.0, [0.0 for i in self.metrics]
                
                for i in range(self.disc_batches_per_epoch):
                    source_input = source.get_batch(self.batch_size)
                    gen_input = self.G.get_input(self.batch_size)
                    batch_out = self.train_discriminator(source_input, gen_input)
                    batch_loss,batch_metrics = batch_out[0],batch_out[1:]
                    d_loss += batch_loss
                    for i in range(len(self.metrics)):
                        d_metrics[i] += batch_metrics[i]
                    
                for i in range(self.gen_batches_per_epoch):
                    gen_input = self.G.get_input(self.batch_size)
                    batch_out = self.train_generator(gen_input)
                    batch_loss,batch_metrics = batch_out[0],batch_out[1:]
                    g_loss += batch_loss
                    for i in range(len(self.metrics)):
                        g_metrics[i] += batch_metrics[i]

                if self.plot:
                    self.gan_plotter.batch_update([d_loss, g_loss, *d_metrics, *g_metrics])
            
            if epoch >= 10 and self.plot:
                self.gan_plotter.log_epoch()
                
            if epoch % printerval == 0:
                preview_seed = self.G.get_input(self.preview_size)
                generated_images = np.array(
                    self.generator.predict(preview_seed))
                self.image_sources[0].save(
                    epoch, generated_images, self.preview_rows, self.preview_cols, self.preview_margin)

    def train_n_eras(self, eras, epochs, printerval, ma_size):
        if self.plot:
            from helpers.GanPlotter import GanPlotter
            self.gan_plotter = GanPlotter(moving_average_size=ma_size, labels=self.plot_labels)
        for i in range(eras):
            self.train(epochs, printerval)
            self.save_generator((i+1)*epochs)