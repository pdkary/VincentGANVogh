from keras.layers import Input, Activation, Cropping2D, Concatenate
from GanBuilder import GanBuilder
from keras.models import Model
import tensorflow as tf
import numpy as np
  
class DCGAN(GanBuilder):
  def __init__(self,gan_shape_config,gan_building_config,gan_training_config):
    super().__init__(gan_shape_config,gan_building_config,gan_training_config)

    self.real_image_input = Input(shape=self.img_shape, name="image_input")
    self.latent_model_input = Input(shape=self.gen_constant_shape, name="latent_space_input")
    self.noise_model_input = Input(shape=self.img_shape,name="noise_image_input")
    self.style_model_input = Input(shape=(self.style_size,),name="style_input")

    self.discriminator_input = [self.real_image_input,
                       self.latent_model_input,
                       self.style_model_input,
                       self.noise_model_input]
    
    self.generator_input = [self.style_model_input,
                            self.noise_model_input,
                            self.latent_model_input]

    self.init_noise_model()
    S = self.build_style_model(self.style_model_input,self.style_layer_size,self.style_layers)
    G = self.build_generator(self.latent_model_input,S)
    D = self.build_discriminator(self.real_image_input)
    
    self.G = Model(inputs=self.generator_input,outputs=G, name="generator_base")
    self.D = Model(inputs=self.real_image_input,outputs=D,name="discriminator_base")

  def init_noise_model(self):
    self.N = Activation('linear')(self.noise_model_input)
    noise_layers = [self.N]
    noise_sizes = [self.N.shape[1]]
    curr_size = self.N.shape[1]
    while curr_size > self.gen_constant_shape[0]:
      curr_size = curr_size//2
      noise_layers.append(Cropping2D(curr_size//2)(noise_layers[-1]))
      noise_sizes.append(curr_size)
    
    self.noise_dict = dict(zip(noise_sizes,noise_layers))

  def set_trainable(self,gen_state,disc_state):
    self.G.trainable = gen_state
    self.D.trainable = disc_state
    for layer in self.G.layers:
      layer.trainable = gen_state
    for layer in self.D.layers:
      layer.trainable = disc_state

  def GenModel(self):
    self.set_trainable(True,False)
    generated_output = self.G(self.generator_input)
    discriminated_output = self.D(generated_output,training=False)
    self.gen_model = Model(inputs=self.generator_input,outputs=discriminated_output,name="generator_model")
    self.gen_model.compile(optimizer=self.gen_optimizer,loss=self.gen_loss_function,metrics=['accuracy'])
    self.gen_model.summary()
    return self.gen_model
  
  def DisModel(self):
    self.set_trainable(False,True)
    generated_imgs = self.G(self.generator_input,training=False)

    d_real = self.D(self.real_image_input)    
    d_fake = self.D(generated_imgs)

    output_arr = [d_real,d_fake]

    self.dis_model = Model(inputs=self.discriminator_input,outputs=output_arr,name="discriminator_model")
    self.dis_model.compile(optimizer=self.disc_optimizer,loss=self.disc_loss_function,metrics=['accuracy'])
    self.dis_model.summary()
    return self.dis_model