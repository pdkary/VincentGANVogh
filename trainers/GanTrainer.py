from typing import List
from keras.layers.convolutional import UpSampling2D
from numpy.lib.function_base import median
from models.GanInput import RealImageInput
from config.TrainingConfig import DataConfig, GanTrainingConfig
from config.DiscriminatorConfig import DiscriminatorModelConfig
from config.GeneratorConfig import GeneratorModelConfig
from models.Generator.Generator import Generator
from models.Discriminator import Discriminator
import numpy as np
import tensorflow as tf

cross_entropy = tf.keras.losses.BinaryCrossentropy()

class GanTrainer(GanTrainingConfig):
  def __init__(self,
               gen_model_config:    GeneratorModelConfig,
               disc_model_config:   DiscriminatorModelConfig,
               gan_training_config: GanTrainingConfig,
               data_configs:         List[DataConfig]):
    GanTrainingConfig.__init__(self,**gan_training_config.__dict__)
    self.preview_size = int(max([d.preview_cols*d.preview_rows for d in data_configs]))
    self.batch_size = int(median([d.batch_size for d in data_configs]))
    self.generator: Generator = Generator(gen_model_config,self.batch_size,self.preview_size)
    self.discriminator: Discriminator = Discriminator(disc_model_config)
    self.image_sources: List[RealImageInput] = [RealImageInput(d) for d in data_configs]
    
    self.GenModel = self.generator.build_generator()
    self.DisModel = self.discriminator.build()
    self.model_output_path = data_configs[0].data_path + "/models"
    
    for source in self.image_sources:
      source.load()
      
  def train_generator(self):
    generator_input = self.generator.get_input()
    with tf.GradientTape() as gen_tape:
      generated_images = self.GenModel(generator_input,training=True)
      fake_out = self.DisModel(generated_images,training=False)
      fake_label = self.gen_label*tf.ones_like(fake_out)
      
      loss = cross_entropy(fake_label,fake_out)
      if self.gen_label == 0:
        acc = 1 - np.average(fake_out)
      else:
        acc = 1 - sum(abs(fake_label - fake_out))/(self.gen_label*len(fake_label))
      
      gradients_of_generator = gen_tape.gradient(loss,self.GenModel.trainable_variables)
      self.generator.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.GenModel.trainable_variables))
    return loss,acc

  def train_discriminator(self,training_images):
    generator_input = self.generator.get_input()
    
    with tf.GradientTape() as disc_tape:
      generated_images = self.GenModel(generator_input,training=False)
      real_out = self.DisModel(training_images,training=True)
      fake_out = self.DisModel(generated_images,training=True)
      real_label = self.disc_labels[0]*tf.ones_like(real_out)
      fake_label = self.disc_labels[1]*tf.ones_like(fake_out)
      
      real_error = abs(real_label - real_out)
      fake_error = abs(fake_label - fake_out)
      real_loss = cross_entropy(real_label, real_out)
      fake_loss = cross_entropy(fake_label, fake_out)
      
      if self.disc_labels[0] == 0:
        real_acc = 1 - np.average(real_out)
      else:
        real_acc = 1 - sum(real_error)/sum(real_label)
        
      
      if self.disc_labels[1] == 0:
        fake_acc = 1 - np.average(fake_out)
      else:
        real_acc = 1 - sum(fake_error)/sum(fake_label)
      
      loss = (real_loss + fake_loss)/2
      acc = (real_acc + fake_acc)/2
      
      gradients_of_discriminator = disc_tape.gradient(loss,self.DisModel.trainable_variables)
      self.discriminator.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.DisModel.trainable_variables))
    return loss,acc
    
  def train(self,epochs,batches_per_epoch,printerval):
    for epoch in range(epochs):
      if self.plot:
        self.gan_plotter.start_batch()
      
      for i in range(batches_per_epoch):
        for source in self.image_sources:
          img_batch = source.get_batch()
          bd_loss,bd_acc = self.train_discriminator(img_batch)
          bg_loss,bg_acc = self.train_generator()
          if self.plot:
            self.gan_plotter.batch_update(bd_loss,bd_acc,bg_loss,bg_acc)
      
      if self.plot: 
        self.gan_plotter.end_batch()
      
      if epoch % printerval == 0:
        preview_seed = self.generator.get_input(training=False)
        generated_images = np.array(self.GenModel.predict(preview_seed))
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
      self.GenModel.save(self.model_output_path + filename)
