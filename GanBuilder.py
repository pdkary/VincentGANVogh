from keras.layers import Dense,Reshape,Conv2D,Flatten
from GanBase import GanBase
from MinibatchDiscrimination import MinibatchDiscrimination
from math import prod

class GanBuilder(GanBase):
  def build_style_model(self,input_tensor,size,layers):
    style_model = self.style_model_block(input_tensor,size)
    for i in range(layers-1):
      style_model = self.style_model_block(style_model,size)
    return style_model

  def build_generator(self,latent_input_tensor,style_model):

    gen_model = Dense(prod(list(self.gen_constant_shape)), kernel_initializer = 'he_normal')(latent_input_tensor)
    gen_model = Reshape(self.gen_constant_shape)(gen_model)
    
    for shape,upsampling,noise in zip(self.gen_layer_shapes,self.gen_layer_upsampling,self.gen_layer_noise):
      gen_model = self.generator_block(gen_model,style_model,*shape,upsampling=upsampling,noise=noise)
    
    gen_model = Conv2D(self.channels, 1, padding='same',activation='sigmoid')(gen_model)
    return gen_model

  def build_discriminator(self,disc_model_input):
    disc_model = disc_model_input
    for shape in self.disc_layer_shapes:
      disc_model = self.disc_conv_block(disc_model,*shape)     
    
    disc_model = Flatten()(disc_model)
    disc_model = MinibatchDiscrimination(self.minibatch_size,self.channels)(disc_model) if self.minibatch else disc_model
    for size in self.disc_dense_sizes:
      disc_model = self.disc_dense_block(disc_model,size)
    disc_model = Dense(1,activation="sigmoid", kernel_initializer = 'he_normal')(disc_model)
    return disc_model