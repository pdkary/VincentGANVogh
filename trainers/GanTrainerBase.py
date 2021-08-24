from abc import ABC
from typing import Tuple
from models.GanInput import RealImageInput
from config.TrainingConfig import DataConfig, GanTrainingConfig
from config.DiscriminatorConfig import DiscriminatorModelConfig
from config.GeneratorConfig import GeneratorModelConfig
from models.Generator.Generator import Generator
from models.Discriminator import Discriminator
import numpy as np
import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(real_output), real_output)
  real_acc = np.average(real_output)
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  fake_acc = 1 - np.average(fake_output)
  total_loss = real_loss + fake_loss
  total_acc = (real_acc + fake_acc)/2
  return total_loss,total_acc
 
def generator_loss(fake_output):
  ones = tf.ones_like(fake_output)
  loss = cross_entropy(ones,fake_output)
  acc = np.average(fake_output)
  return loss,acc

class GanTrainer(GanTrainingConfig, ABC):
  def __init__(self,
               gen_model_config:    GeneratorModelConfig,
               disc_model_config:   DiscriminatorModelConfig,
               gan_training_config: GanTrainingConfig,
               data_config:         DataConfig):
    GanTrainingConfig.__init__(self,**gan_training_config.__dict__)
    self.preview_size = data_config.preview_cols*data_config.preview_rows
    self.generator: Generator = Generator(gen_model_config,data_config.batch_size,self.preview_size)
    self.discriminator: Discriminator = Discriminator(disc_model_config)
    self.image_source: RealImageInput = RealImageInput(data_config)

    self.GenModel = self.generator.build_generator()
    self.DisModel = self.discriminator.build()
    self.model_output_path = data_config.data_path + "/models"
    
    self.image_source.load()
      
  def train_generator(self) -> Tuple[float,float]:
      pass

  def train_discriminator(self,training_images) -> Tuple[float,float]:
      pass
    
  def train(self,epochs,batches_per_epoch,printerval):
    for epoch in range(epochs):
      if self.plot:
        self.gan_plotter.start_batch()
      
      for i in range(batches_per_epoch):
        img_batch = self.image_source.get_batch()
        bd_loss,bd_acc = self.train_discriminator(img_batch)
        bg_loss,bg_acc = self.train_generator()
        if self.plot:
          self.gan_plotter.batch_update(bd_loss,bd_acc,bg_loss,bg_acc)
      
      if self.plot: 
        self.gan_plotter.end_batch()
      
      if epoch % printerval == 0:
        preview_seed = self.generator.get_input(training=False)
        generated_images = np.array(self.GenModel.predict(preview_seed))
        self.image_source.save(epoch,generated_images)

      if epoch >= 10 and self.plot:
        self.gan_plotter.log_epoch()
  
  def train_n_eras(self,eras,epochs,batches_per_epoch,printerval,ma_size):
    if self.plot:
      from helpers.GanPlotter import GanPlotter
      self.gan_plotter = GanPlotter(moving_average_size=ma_size)
    for i in range(eras):
      self.train(epochs,batches_per_epoch,printerval)
      filename = self.image_source.data_helper.model_name + "%d"%((i+1)*epochs)
      self.GenModel.save(self.model_output_path + filename)
