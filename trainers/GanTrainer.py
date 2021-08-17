from GanConfig import DiscriminatorModelConfig, GanTrainingConfig, GeneratorModelConfig, NoiseModelConfig, StyleModelConfig
from Generator import Generator
from Discriminator import Discriminator
from helpers.DataHelper import DataHelper
from helpers.GanPlotter import GanPlotter
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

class GanTrainer(GanTrainingConfig):
  def __init__(self,
               gen_model_config:    GeneratorModelConfig,
               noise_model_config:  NoiseModelConfig,
               style_model_config:  StyleModelConfig,
               disc_model_config:   DiscriminatorModelConfig,
               gan_training_config: GanTrainingConfig):
    GanTrainingConfig.__init__(self,**gan_training_config.__dict__)
    self.generator = Generator(gen_model_config,noise_model_config,style_model_config)
    self.discriminator = Discriminator(disc_model_config)

    self.GenModel = self.generator.build_generator()
    self.DisModel = self.discriminator.build()

    self.img_shape = gen_model_config.img_shape
    self.preview_margin = 16
    self.preview_size = self.preview_rows*self.preview_cols

    self.image_output_path = self.data_path + "/images"
    self.model_output_path = self.data_path + "/models"
    
    print("Preparing Dataset".upper())
    self.images = DataHelper.load_data(self.data_path,self.img_shape,self.image_type,self.flip_lr,self.load_n_percent)
    self.dataset = tf.data.Dataset.from_tensor_slices(self.images).batch(self.batch_size)
    self.dataset_size = len(self.images)
    print("DATASET LOADED")
      
  def train_generator(self):
    generator_input = self.generator.get_input(self.batch_size)
    with tf.GradientTape() as gen_tape:
      generated_images = self.GenModel(generator_input,training=True)
      discriminated_gens = self.DisModel(generated_images,training=False)
      
      g_loss,g_acc = generator_loss(discriminated_gens)
      gradients_of_generator = gen_tape.gradient(g_loss,self.GenModel.trainable_variables)
      self.generator.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.GenModel.trainable_variables))
    return g_loss,g_acc

  def train_discriminator(self,training_imgs):
    generator_input = self.generator.get_input(self.batch_size)
    with tf.GradientTape() as disc_tape:
      generated_images = self.GenModel(generator_input,training=False)
      real_out = self.DisModel(training_imgs,training=True)
      fake_out = self.DisModel(generated_images,training=True)
      
      d_loss,d_acc = discriminator_loss(real_out,fake_out)
      gradients_of_discriminator = disc_tape.gradient(d_loss,self.DisModel.trainable_variables)
      self.generator.gen_optimizer.apply_gradients(zip(gradients_of_discriminator, self.DisModel.trainable_variables))
    return d_loss,d_acc
    
  
  def train(self,epochs,batches_per_epoch,printerval):
    self.dataset = self.dataset.shuffle(self.dataset_size//8)
    for epoch in range(epochs):
      self.gan_plotter.start_batch()
      
      for img_batch in self.dataset.take(batches_per_epoch):
        bd_loss,bd_acc = self.train_discriminator(img_batch)
        bg_loss,bg_acc = self.train_generator()
        self.gan_plotter.batch_update(bd_loss,bd_acc,bg_loss,bg_acc)
        
      self.gan_plotter.end_batch()
      
      if epoch % printerval == 0:
        preview_seed = self.generator.get_input(self.preview_size)
        generated_images = np.array(self.GenModel.predict(preview_seed))
        DataHelper.save_images(epoch,generated_images,self.img_shape,self.preview_rows,self.preview_cols,self.preview_margin,self.image_output_path,self.image_type)

      if epoch >= 10:
        self.gan_plotter.log_epoch()
  
  def train_n_eras(self,eras,epochs,batches_per_epoch,printerval,ma_size):
    self.gan_plotter = GanPlotter(moving_average_size=ma_size)
    for i in range(eras):
      self.train(epochs,batches_per_epoch,printerval)
      filename = self.model_name + "%d"%((i+1)*epochs)
      self.GenModel.save(self.model_output_path + filename)
