from typing import List
from models.GanInput import RealImageInput
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
    self.batch_ratio = self.gen_batch_size // self.disc_batch_size
    
    self.generator = self.G.build()
    self.discriminator = self.D.build()
    self.model_output_path = data_configs[0].data_path + "/models"
    
    for source in self.image_sources:
      source.load()
      
  def train_generator(self):
    d = self.disc_batch_size
    g = self.gen_batch_size
    generator_input = self.G.get_input(g)
    with tf.GradientTape() as gen_tape:
      generated_images = self.generator(generator_input,training=True)
      fake_out = np.zeros(shape=(self.disc_batch_size,*self.G.img_shape))
      
      for i in range(self.batch_ratio):
        fake_out[i*d:i*d+d] = self.discriminator(generated_images[i*d:i*d+d],training=False)
      
      fake_label = self.gen_label*tf.ones_like(fake_out)
      loss = self.G.loss_function(fake_label,fake_out)
      g_avg = np.average(fake_out)
      
      gradients_of_generator = gen_tape.gradient(loss,self.generator.trainable_variables)
      self.G.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
    return loss,g_avg

  def train_discriminator(self,training_images):
    generator_input = self.G.get_input(self.disc_batch_size)
    
    with tf.GradientTape() as disc_tape:
      generated_images = self.generator(generator_input,training=False)
      real_out = self.discriminator(training_images,training=True)
      fake_out = self.discriminator(generated_images,training=True)
      real_label = self.disc_labels[0]*tf.ones_like(real_out)
      fake_label = self.disc_labels[1]*tf.ones_like(fake_out)
      
      real_loss = self.D.loss_function(real_label, real_out)
      fake_loss = self.D.loss_function(fake_label, fake_out)
      
      loss = (real_loss + fake_loss)/2
      d_avg = np.average(real_out)
      g_avg = np.average(fake_out)      
      gradients_of_discriminator = disc_tape.gradient(loss,self.discriminator.trainable_variables)
      self.D.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    return loss,d_avg,g_avg
    
  def train(self,epochs,batches_per_epoch,printerval):
    for epoch in range(epochs):
      if self.plot:
        self.gan_plotter.start_batch()
      
      for i in range(batches_per_epoch):
        for source in self.image_sources:
          disc_batch = source.get_batch(self.disc_batch_size)
          bd_loss,bd_avg,bg_avg = self.train_discriminator(disc_batch)
          bg_loss,bg_avg = self.train_generator()
          if self.plot:
            self.gan_plotter.batch_update(bd_loss,bd_avg,bg_avg,bg_loss,bg_avg)
      
      if self.plot: 
        self.gan_plotter.end_batch()
      
      if epoch % printerval == 0:
        preview_seed = self.G.get_input(self.preview_size)
        generated_images = np.array(self.generator.predict(preview_seed))
        self.image_sources[0].save(epoch,generated_images)

      if epoch >= 10 and self.plot:
        self.gan_plotter.log_epoch()
  
  def train_n_eras(self,eras,epochs,batches_per_epoch,printerval,ma_size):
    if self.plot:
      from helpers.GanPlotter import GanPlotter
      self.gan_plotter = GanPlotter(moving_average_size=ma_size)
    for i in range(eras):
      self.train(epochs,batches_per_epoch,printerval)
      filename = self.image_sources[0].data_helper.model_name + "%d"%((i+1)*epochs)
      print(self.model_output_path + filename)
      self.generator.save(self.model_output_path + filename)
