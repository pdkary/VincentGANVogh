from typing import List
from layers.GanInput import RealImageInput
from config.TrainingConfig import DataConfig, GanTrainingConfig
from config.DiscriminatorConfig import DiscriminatorModelConfig
from config.GeneratorConfig import GeneratorModelConfig
from models.Generator import Generator
from models.Discriminator import Discriminator
import numpy as np
import tensorflow as tf

class GenTapeTrainer(GanTrainingConfig):
  def __init__(self,
               gen_model_config:    GeneratorModelConfig,
               disc_model_config:   DiscriminatorModelConfig,
               gan_training_config: GanTrainingConfig,
               data_configs:         List[DataConfig]):
    GanTrainingConfig.__init__(self,**gan_training_config.__dict__)
    self.G: Generator = Generator(gen_model_config)
    self.D: Discriminator = Discriminator(disc_model_config)
    self.image_sources: List[RealImageInput] = [RealImageInput(d) for d in data_configs]
    self.preview_size = self.preview_cols*self.preview_rows
    
    label_shape = (self.batch_size,self.D.output_dim)
    self.real_label = self.disc_labels[0]*tf.ones(shape=label_shape)
    self.fake_label = self.disc_labels[1]*tf.ones(shape=label_shape)
    self.gen_label = self.gen_label*tf.ones(shape=label_shape)
    
    self.generator = self.G.build()
    self.discriminator = self.D.build()
    self.model_output_path = data_configs[0].data_path + "/models"
    
    for source in self.image_sources:
      source.load()
      
  def train_generator(self,gen_input):
    with tf.GradientTape() as gen_tape:
      generated_images = self.generator(gen_input,training=False)
      fake_out = self.discriminator(generated_images,training=True)
      g_loss = self.G.loss_function(self.gen_label, fake_out)
      g_avg = np.mean(fake_out)
      gradients_of_generator = gen_tape.gradient(g_loss,self.generator.trainable_variables)
      self.G.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
    return g_loss,g_avg
  
  def train_discriminator(self,disc_input,gen_input):
    with tf.GradientTape() as disc_tape:
      generated_images = self.generator(gen_input,training=False)
      real_out = self.discriminator(disc_input,training=True)
      fake_out = self.discriminator(generated_images,training=True)
      
      real_loss = self.D.loss_function(self.real_label, real_out)
      fake_loss = self.D.loss_function(self.fake_label, fake_out)
      d_loss = (real_loss + fake_loss)/2
      d_avg = np.mean(real_out)
      
      gradients_of_discriminator = disc_tape.gradient(d_loss,self.discriminator.trainable_variables)
      self.D.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    return d_loss,d_avg
    
  def train(self,epochs,batches_per_epoch,printerval):
    for epoch in range(epochs):
      if self.plot:
        self.gan_plotter.start_epoch()
      
      for i in range(batches_per_epoch):
        for source in self.image_sources:
          d_loss,d_avg = 0,0
          g_loss,g_avg = 0,0
          for i in range(self.disc_batches_per_epoch):
            disc_input = source.get_batch(self.batch_size)
            gen_input = self.G.get_input(self.batch_size)
            batch_loss,batch_avg = self.train_discriminator(disc_input,gen_input)
            d_loss += batch_loss
            d_avg += batch_avg
          for i in range(self.gen_batches_per_epoch):
            gen_input = self.G.get_input(self.batch_size)
            batch_loss,batch_avg = self.train_generator(gen_input)
            g_loss += batch_loss
            g_avg += batch_avg
          
          if self.plot:
            d_loss /= self.disc_batches_per_epoch
            d_avg /= self.disc_batches_per_epoch
            g_loss /= self.gen_batches_per_epoch
            g_avg /= self.gen_batches_per_epoch
            self.gan_plotter.batch_update([d_loss,d_avg,g_avg,g_loss])
      
      if epoch % printerval == 0:
        preview_seed = self.G.get_input(self.preview_size)
        generated_images = np.array(self.generator.predict(preview_seed))
        self.image_sources[0].save(epoch,generated_images,self.preview_rows,self.preview_cols,self.preview_margin)

      if epoch >= 10 and self.plot:
        self.gan_plotter.log_epoch()
  
  def train_n_eras(self,eras,epochs,batches_per_epoch,printerval,ma_size):
    if self.plot:
      from helpers.GanPlotter import GanPlotter
      self.gan_plotter = GanPlotter(moving_average_size=ma_size,labels=["D_loss","D_avg","G_avg","G_Loss"])
    for i in range(eras):
      self.train(epochs,batches_per_epoch,printerval)
      filename = self.image_sources[0].data_helper.model_name + "%d"%((i+1)*epochs)
      print(self.model_output_path + filename)
      self.generator.save(self.model_output_path + filename)
