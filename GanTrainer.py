from GanConfig import GanTrainingConfig
from Generator import Generator
from Discriminator import Discriminator
from DataHelper import DataHelper
from GanPlotter import GanPlotter
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
  def __init__(self,gen_model_config,noise_model_config,style_model_config,disc_model_config,gan_training_config):
    GanTrainingConfig.__init__(self,**gan_training_config.__dict__)
    self.generator = Generator(gen_model_config,noise_model_config,style_model_config)
    self.discriminator = Discriminator(disc_model_config)

    self.GenModel = self.generator.build()
    self.DisModel = self.discriminator.build()

    self.img_shape = gen_model_config.img_shape
    self.noise_image_size = noise_model_config.noise_image_size
    self.style_latent_size = style_model_config.style_latent_size
    self.preview_margin = 16
    self.preview_size = self.preview_rows*self.preview_cols

    self.image_output_path = self.data_path + "/images"
    self.model_output_path = self.data_path + "/models"
    
    self.gen_constant_shape = gen_model_config.gen_constant_shape
    self.gen_constant = tf.random.normal(shape=self.gen_constant_shape)
    
    self.training_latent = self.get_batched_constant(self.batch_size)
    self.preview_latent = self.get_batched_constant(self.preview_size)

    print("Preparing Dataset".upper())
    self.images = DataHelper.load_data(self.data_path,
                                       self.img_shape,
                                       self.image_type,
                                       self.flip_lr,
                                       self.load_n_percent)
    self.dataset = tf.data.Dataset.from_tensor_slices(self.images).batch(self.batch_size)
    print("DATASET LOADED")

  def get_batched_constant(self,batch_size):
    gc_batch = np.full((batch_size,*self.gen_constant_shape),0.0,dtype=np.float32)
    for i in range(batch_size):
      gc_batch[i] = self.gen_constant
    return gc_batch

  #Style Z and latent noise
  def style_noise(self,batch_size):
    return tf.random.normal(shape = (batch_size,self.style_latent_size))

  #Noise Sample
  def noise(self,batch_size):
    noise_batch = np.full((batch_size,*self.noise_image_size),0.0,dtype=np.float32)
    for i in range(batch_size):
      noise_batch[i] = tf.random.normal(shape=self.noise_image_size,stddev=self.gauss_factor)
    return noise_batch
  
  def get_generator_input(self,training=True):
    batch_size = self.batch_size if training else self.preview_size
    latent_noise = self.training_latent if training else self.preview_latent
    return [latent_noise,
            self.style_noise(batch_size),
            self.noise(batch_size)]
    
  def train_step(self,training_imgs):
    generator_input = self.get_generator_input()
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = self.GenModel(generator_input, training=True)
      real_output = self.DisModel(training_imgs,training=True)
      fake_output = self.DisModel(generated_images,training=True)
      
      g_loss,g_acc = generator_loss(fake_output)
      d_loss,d_acc = discriminator_loss(real_output,fake_output)
        
      gradients_of_generator = gen_tape.gradient(g_loss, self.GenModel.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(d_loss, self.DisModel.trainable_variables)
  
      self.generator.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.GenModel.trainable_variables))
      self.discriminator.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.DisModel.trainable_variables))
  
    return d_loss,d_acc,g_loss,g_acc
  
  def train(self,epochs,batches_per_epoch,printerval):
    for epoch in range(epochs):
      self.gan_plotter.start_batch()
      
      for img_batch in self.dataset.take(batches_per_epoch):
        bd_loss,bd_acc,bg_loss,bg_acc = self.train_step(img_batch)
        self.gan_plotter.batch_update(bd_loss,bd_acc,bg_loss,bg_acc)
        
      self.gan_plotter.end_batch()
      
      if epoch % printerval == 0:
        preview_seed = self.get_generator_input(training=False)
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
