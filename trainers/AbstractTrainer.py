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
        self.image_sources: List[RealImageInput] = [
            RealImageInput(d) for d in data_configs]
        self.preview_size = self.preview_cols*self.preview_rows

        label_shape = (self.batch_size, self.D.output_dim)
        self.real_label = self.disc_labels[0]*tf.ones(shape=label_shape)
        self.fake_label = self.disc_labels[1]*tf.ones(shape=label_shape)
        self.gen_label = self.gen_label*tf.ones(shape=label_shape)

        self.generator = self.G.build()
        self.discriminator = self.D.build()
        self.model_output_path = data_configs[0].data_path + "/models"

        for source in self.image_sources:
            source.load()

    @abstractmethod
    def train_generator(self, gen_input):
        return 0.0, 0.0

    @abstractmethod
    def train_discriminator(self, source_input, gen_input):
        return 0.0, 0.0

    def train(self, epochs, printerval):
        for epoch in range(epochs):
            if self.plot:
                self.gan_plotter.start_epoch()

            for source in self.image_sources:
                d_loss, d_avg = 0, 0
                g_loss, g_avg = 0, 0
                for i in range(self.disc_batches_per_epoch):
                    source_input = source.get_batch(self.batch_size)
                    gen_input = self.G.get_input(self.batch_size)
                    batch_loss, batch_avg = self.train_discriminator(source_input, gen_input)
                    d_loss += batch_loss
                    d_avg += batch_avg
                for i in range(self.gen_batches_per_epoch):
                    gen_input = self.G.get_input(self.batch_size)
                    batch_loss, batch_avg = self.train_generator(gen_input)
                    g_loss += batch_loss
                    g_avg += batch_avg

                if self.plot:
                    d_loss /= self.disc_batches_per_epoch
                    d_avg /= self.disc_batches_per_epoch
                    g_loss /= self.gen_batches_per_epoch
                    g_avg /= self.gen_batches_per_epoch
                    self.gan_plotter.batch_update(
                        [d_loss, d_avg, g_avg, g_loss])
            
            if epoch >= 10 and self.plot:
                self.gan_plotter.log_epoch()
                
            if epoch % printerval == 0:
                preview_seed = self.G.get_input(self.preview_size)
                generated_images = np.array(
                    self.generator.predict(preview_seed))
                self.image_sources[0].save(
                    epoch, generated_images, self.preview_rows, self.preview_cols, self.preview_margin)

    def train_n_eras(self, eras, epochs, batches_per_epoch, printerval, ma_size):
        if self.plot:
            from helpers.GanPlotter import GanPlotter
            self.gan_plotter = GanPlotter(moving_average_size=ma_size, labels=[
                                          "D_loss", "D_avg", "G_avg", "G_Loss"])
        for i in range(eras):
            self.train(epochs, batches_per_epoch, printerval)
            filename = self.image_sources[0].data_helper.model_name + \
                "%d" % ((i+1)*epochs)
            print(self.model_output_path + filename)
            self.generator.save(self.model_output_path + filename)
