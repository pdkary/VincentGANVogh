from DCGAN import DCGAN
from DataHelper import DataHelper
import numpy as np
import tensorflow as tf
from jupyterplot import ProgressPlot
import time

class GanTrainer(DCGAN):
  def __init__(self,gan_shape_config,gan_building_config,gan_training_config):
    super().__init__(gan_shape_config,gan_building_config,gan_training_config)

    self.preview_margin = 16

    self.GenModel = self.GenModel()
    self.DisModel = self.DisModel()

    self.image_output_path = self.data_path + "/images"
    self.model_output_path = self.data_path + "/models"

    self.preview_size = self.preview_rows*self.preview_cols

    self.gen_constant = tf.random.normal(shape=self.gen_constant_shape)
    
    self.training_latent = self.get_batched_constant(self.batch_size)
    self.preview_latent = self.get_batched_constant(self.preview_size)

    self.ones = np.ones((self.batch_size, 1), dtype=np.float32)
    self.zeros = np.zeros((self.batch_size, 1), dtype=np.float32)
    
    print("Preparing Dataset".upper())
    self.images = DataHelper.load_data(self.data_path,self.img_shape,self.image_type,self.flip_lr,self.load_n_percent)
    self.dataset = tf.data.Dataset.from_tensor_slices(self.images).batch(self.batch_size)
    print("DATASET LOADED")

  
  def get_batched_constant(self,batch_size):
    gc_batch = np.full((batch_size,*self.gen_constant_shape),0.0,dtype=np.float32)
    for i in range(batch_size):
      gc_batch[i] = self.gen_constant
    return gc_batch

  #Style Z and latent noise
  def style_noise(self,batch_size):
    return tf.random.normal(shape = (batch_size,self.style_size))
  
  def homogenous_noise(self,batch_size):
    homo_ones = tf.ones(shape=(batch_size,*self.img_shape))
    scale = tf.random.uniform((1,1),minval=0,maxval=1,dtype=tf.float32)
    return scale*homo_ones

  #Noise Sample
  def noiseImage(self,batch_size):
    noise_batch = np.full((batch_size,*self.img_shape),0.0,dtype=np.float32)
    for i in range(batch_size):
      n_image = tf.random.normal(shape=self.img_shape,stddev=self.gauss_factor)
      noise_batch[i] = n_image
    return noise_batch
  
  def get_generator_input(self,training=True):
    batch_size = self.batch_size if training else self.preview_size
    latent_noise = self.training_latent if training else self.preview_latent
    return [self.style_noise(batch_size),
            self.noiseImage(batch_size),
            latent_noise]
    
  def get_discriminator_input(self,training_imgs):
    return [training_imgs,
            self.training_latent,
            self.style_noise(self.batch_size),
            self.noiseImage(self.batch_size)]

  def train_generator(self, noise_data):
    self.set_trainable(True,False)
    g_losses = self.GenModel.train_on_batch(noise_data,self.ones)
    return g_losses[0],g_losses[1]
  
  def train_discriminator(self,training_data):
    self.set_trainable(False,True)
    ## disc outputs are [real, fake, noise, homogenous]
    d_losses = self.DisModel.train_on_batch(training_data,[self.ones,self.zeros])
    label = self.DisModel.metrics_names.index('discriminator_base_accuracy')
    return d_losses[0],d_losses[label]
    
  def train_step(self,training_imgs):
    generator_input = self.get_generator_input()
    disc_input = self.get_discriminator_input(training_imgs)
    g_loss,g_acc = self.train_generator(generator_input)
    d_loss,d_acc = self.train_discriminator(disc_input)
    return d_loss,d_acc,g_loss,g_acc
  
  def train(self,epochs,batches_per_epoch,printerval,ma_size):
    d_loss_ma_buffer, g_loss_ma_buffer = [], []
    d_acc_ma_buffer, g_acc_ma_buffer = [], []
    time_ma_buffer = []
    
    for epoch in range(epochs):
      epoch_start = time.time()
      batch_d_loss, batch_g_loss = [], []
      batch_d_acc, batch_g_acc = [], []
      
      for img_batch in self.dataset.take(batches_per_epoch):
        bd_loss,bd_acc,bg_loss,bg_acc = self.train_step(img_batch)
        batch_d_loss.append(bd_loss)
        batch_g_loss.append(bg_loss)
        batch_d_acc.append(bd_acc)
        batch_g_acc.append(bg_acc)
      
      d_loss, g_loss = np.mean(batch_d_loss),np.mean(batch_g_loss)
      d_acc, g_acc = np.mean(batch_d_acc), np.mean(batch_g_acc)

      if epoch % printerval == 0:
        preview_seed = self.get_generator_input(training=False)
        generated_images = np.array(self.G.predict(preview_seed))
        DataHelper.save_images(epoch,generated_images,self.img_shape,self.preview_rows,self.preview_cols,self.preview_margin,self.image_output_path,self.image_type)

      epoch_elapsed = time.time()-epoch_start

      if epoch >= 10:
        d_loss_ma_buffer.append(d_loss)
        g_loss_ma_buffer.append(g_loss)
        d_acc_ma_buffer.append(d_acc)
        g_acc_ma_buffer.append(g_acc)
        time_ma_buffer.append(epoch_elapsed)

        d_loss_ma_buffer = d_loss_ma_buffer[1:] if len(d_loss_ma_buffer) >= ma_size else d_loss_ma_buffer
        g_loss_ma_buffer = g_loss_ma_buffer[1:] if len(g_loss_ma_buffer) >= ma_size else g_loss_ma_buffer
        d_acc_ma_buffer = d_acc_ma_buffer[1:] if len(d_acc_ma_buffer) >= ma_size else d_acc_ma_buffer
        g_acc_ma_buffer = g_acc_ma_buffer[1:] if len(g_acc_ma_buffer) >= ma_size else g_acc_ma_buffer
        time_ma_buffer = time_ma_buffer[1:] if len(time_ma_buffer) >= ma_size else time_ma_buffer

        d_loss_ma,g_loss_ma = np.mean(d_loss_ma_buffer),np.mean(g_loss_ma_buffer)
        d_acc_ma,g_acc_ma = np.mean(d_acc_ma_buffer), np.mean(g_acc_ma_buffer)
        time_ma = np.mean(time_ma_buffer)

        self.progress_plot.update([[d_loss,d_loss_ma],[d_acc,d_acc_ma],[g_loss,g_loss_ma],[g_acc,g_acc_ma],[epoch_elapsed,time_ma]])
  
  def train_n_eras(self,eras,epochs,batches_per_epoch,printerval,ma_size):
    self.progress_plot = ProgressPlot(plot_names =['D Loss','D acc','G Loss','G acc', 'Epoch Duration'],line_names=["value", "MA"])
    for i in range(eras):
      self.train(epochs,batches_per_epoch,printerval,ma_size)
      filename = self.model_name + "%d"%((i+1)*epochs)
      self.G.save(self.model_output_path + filename)
