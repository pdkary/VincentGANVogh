from GanConfig import GanTrainingConfig
from Generator import Generator
from Discriminator import Discriminator
from keras.models import Model
  
class DCGAN(GanTrainingConfig):
  def __init__(self,gen_model_config,noise_model_config,style_model_config,disc_model_config,gan_training_config):
    self.generator = Generator(gen_model_config,noise_model_config,style_model_config)
    self.discriminator = Discriminator(disc_model_config)
    GanTrainingConfig.__init__(self,**gan_training_config.__dict__)

    G = self.generator.build()
    D = self.discriminator.build()
    
    self.G = Model(inputs=self.generator.input,outputs=G, name="generator_base")
    self.D = Model(inputs=self.discriminator.input,outputs=D,name="discriminator_base")

  def set_trainable(self,gen_state,disc_state):
    self.G.trainable = gen_state
    self.D.trainable = disc_state
    for layer in self.G.layers:
      layer.trainable = gen_state
    for layer in self.D.layers:
      layer.trainable = disc_state

  def GenModel(self):
    self.set_trainable(True,False)
    generated_output = self.G(self.generator.input)
    discriminated_output = self.D(generated_output,training=False)
    self.gen_model = Model(inputs=self.generator.input,
                           outputs=discriminated_output,
                           name="generator_model")
    self.gen_model.compile(optimizer=self.gen_optimizer,
                           loss=self.gen_loss_function,
                           metrics=['accuracy'])
    self.gen_model.summary()
    return self.gen_model
  
  def DisModel(self):
    dis_model_input = [self.discriminator.input,*self.generator.input]
    self.set_trainable(False,True)
    generated_imgs = self.G(self.generator.input,training=False)

    d_real = self.D(self.discriminator.input)    
    d_fake = self.D(generated_imgs)

    output_arr = [d_real,d_fake]

    self.dis_model = Model(inputs=dis_model_input,
                           outputs=output_arr,
                           name="discriminator_model")
    self.dis_model.compile(optimizer=self.disc_optimizer,
                           loss=self.disc_loss_function,
                           metrics=['accuracy'])
    self.dis_model.summary()
    return self.dis_model