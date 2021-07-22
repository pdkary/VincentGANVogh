from keras.layers import Input, Activation, Cropping2D
from keras.optimizers import Adam
from GanBuilder import GanBuilder
from keras.models import Model

class GanBuildingConfig():
  def __init__(self,learning_rate,img_shape,latent_size,style_size,kernel_size,relu_alpha,dropout_rate,batch_norm_momentum):
    self.learning_rate = learning_rate
    self.img_shape = img_shape
    self.latent_size = latent_size
    self.style_size = style_size
    self.kernel_size = kernel_size
    self.relu_alpha = relu_alpha
    self.dropout_rate = dropout_rate
    self.batch_norm_momentum = batch_norm_momentum

class DCGAN():
  def __init__(self,gan_building_config):
    self.S = None
    self.N = None
    
    self.img_shape = gan_building_config.img_shape
    self.channels = gan_building_config.img_shape[-1]
    self.learning_rate = gan_building_config.learning_rate
    self.latent_size = gan_building_config.latent_size
    self.style_size = gan_building_config.style_size
    self.kernel_size = gan_building_config.kernel_size
    self.relu_alpha = gan_building_config.relu_alpha
    self.dropout_rate = gan_building_config.dropout_rate
    self.batch_norm_momentum = gan_building_config.batch_norm_momentum

    self.real_image_input = Input(shape=self.img_shape, name="image_input")
    self.latent_model_input = Input(shape=self.latent_size, name="latent_space_input")
    self.noise_model_input = Input(shape=self.img_shape,name="noise_image_input")
    self.style_model_input = Input(shape=(self.style_size,),name="style_input")

    self.ad_optimizer = Adam(self.learning_rate, beta_1 = 0, beta_2 = 0.99, decay = 0.00001)
    self.dis_optimizer = Adam(self.learning_rate, beta_1 = 0, beta_2 = 0.99, decay = 0.00001)

    self.noisy_input = [self.style_model_input,self.noise_model_input,self.latent_model_input]
    self.full_input = [self.real_image_input,*self.noisy_input]

    self.init_noise_model()
    self.GanBuilder = GanBuilder(self.img_shape,self.kernel_size,self.relu_alpha,self.dropout_rate,self.batch_norm_momentum,self.noise_dict)
    self.init_style_model()
    self.init_generator()
    self.init_discriminator()

  def init_generator(self):
    G = self.GanBuilder.build_generator(self.latent_model_input,self.S)
    self.G = Model(inputs=self.noisy_input,outputs=G, name="generator_base")
    
  def init_discriminator(self):
    D = self.GanBuilder.build_discriminator(self.real_image_input)
    self.D = Model(inputs=self.real_image_input,outputs=D,name="discriminator_base")
  
  def init_style_model(self):
    self.S = self.GanBuilder.build_style_model(self.style_model_input,64,3)
  
  def init_noise_model(self):
    self.N = Activation('linear')(self.noise_model_input)
    noise_layers = [self.N]
    noise_sizes = [self.N.shape[1]]
    curr_size = self.N.shape[1]
    while curr_size > 4:
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
    generated_output = self.G(self.noisy_input)
    discriminated_output = self.D(generated_output,training=False)
    self.gen_model = Model(inputs=self.noisy_input,outputs=discriminated_output,name="generator_model")
    self.gen_model.compile(optimizer=self.ad_optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    self.gen_model.summary()
    return self.gen_model
  
  def DisModel(self):
    self.set_trainable(False,True)
    d_real = self.D(self.real_image_input)
    generated_imgs = self.G(self.noisy_input)
    d_fake = self.D(generated_imgs)

    self.dis_model = Model(inputs=self.full_input,outputs=[d_real,d_fake],name="discriminator_model")
    self.dis_model.compile(optimizer=self.dis_optimizer,loss=['binary_crossentropy','binary_crossentropy','binary_crossentropy'],metrics=['accuracy'])
    self.dis_model.summary()
    return self.dis_model