from GanConfig import GanTrainingConfig
from DCGAN import DCGAN
from DataHelper import DataHelper
from GanPlotter import GanPlotter
import numpy as np
import tensorflow as tf

class GanTrainer(DCGAN):
  def __init__(self,gen_model_config,noise_model_config,style_model_config,disc_model_config,gan_training_config):
    DCGAN.__init__(self,gen_model_config,noise_model_config,style_model_config,disc_model_config,gan_training_config)

    self.img_shape = gen_model_config.img_shape
    self.noise_latent_size = noise_model_config.noise_latent_size
    self.style_latent_size = style_model_config.style_latent_size
    self.preview_margin = 16
    self.preview_size = self.preview_rows*self.preview_cols

    self.GenModel = self.GenModel()
    self.DisModel = self.DisModel()

    self.image_output_path = self.data_path + "/images"
    self.model_output_path = self.data_path + "/models"
    self.gen_constant_shape = gen_model_config.gen_constant_shape
    self.gen_constant = tf.random.normal(shape=self.gen_constant_shape)
    
    self.training_latent = self.get_batched_constant(self.batch_size)
    self.preview_latent = self.get_batched_constant(self.preview_size)

    self.ones = np.ones((self.batch_size, 1), dtype=np.float32)
    self.zeros = np.zeros((self.batch_size, 1), dtype=np.float32)
    
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
    noise_batch = np.full((batch_size,self.noise_latent_size),0.0,dtype=np.float32)
    for i in range(batch_size):
      n_image = tf.random.normal(shape=(1,self.noise_latent_size),stddev=self.gauss_factor)
      noise_batch[i] = n_image
    return noise_batch
  
  def get_generator_input(self,training=True):
    batch_size = self.batch_size if training else self.preview_size
    latent_noise = self.training_latent if training else self.preview_latent
    return [latent_noise,
            self.style_noise(batch_size),
            self.noise(batch_size)]
    
  def get_discriminator_input(self,training_imgs):
    return [training_imgs,
            self.training_latent,
            self.style_noise(self.batch_size),
            self.noise(self.batch_size)]

  def train_generator(self, noise_data):
    self.set_trainable(True,False)
    g_losses = self.GenModel.train_on_batch(noise_data,self.ones)
    return g_losses[0],g_losses[1]
  
  def train_discriminator(self,training_data):
    self.set_trainable(False,True)
    d_losses = self.DisModel.train_on_batch(training_data,[self.ones,self.zeros])
    label = self.DisModel.metrics_names.index('discriminator_base_accuracy')
    return d_losses[0],d_losses[label]
    
  def train_step(self,training_imgs):
    generator_input = self.get_generator_input()
    disc_input = self.get_discriminator_input(training_imgs)
    g_loss,g_acc = self.train_generator(generator_input)
    d_loss,d_acc = self.train_discriminator(disc_input)
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
        generated_images = np.array(self.G.predict(preview_seed))
        DataHelper.save_images(epoch,generated_images,self.img_shape,self.preview_rows,self.preview_cols,self.preview_margin,self.image_output_path,self.image_type)

      if epoch >= 10:
        self.gan_plotter.log_epoch()
  
  def train_n_eras(self,eras,epochs,batches_per_epoch,printerval,ma_size):
    self.gan_plotter = GanPlotter(moving_average_size=ma_size)
    for i in range(eras):
      self.train(epochs,batches_per_epoch,printerval)
      filename = self.model_name + "%d"%((i+1)*epochs)
      self.G.save(self.model_output_path + filename)
