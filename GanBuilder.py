from keras.layers import Dense,Reshape,Conv2D,Flatten
from GanBuildingBlocks import GanBuildingBlocks

class GanBuilder(GanBuildingBlocks):
  def __init__(self,img_shape,kernel_size,relu_alpha,dropout_rate,batch_norm_momentum,noise_dict):
    super().__init__(img_shape,kernel_size,relu_alpha,dropout_rate,batch_norm_momentum,noise_dict)

  def build_style_model(self,input_tensor,size,layers):
    style_model = self.style_model_block(input_tensor,size)
    for i in range(layers-1):
      style_model = self.style_model_block(style_model,size)
    return style_model

  def build_generator(self,latent_input_tensor,style_model,filename=None):
    if filename is not None:
      return tf.keras.models.load_model(filename)
    else:
      gen_model = Dense(4*4*1024, kernel_initializer = 'he_normal')(latent_input_tensor)
      gen_model = Reshape((4,4,1024))(gen_model)                                   
      gen_model = self.generator_block(gen_model,style_model,512,3)
      gen_model = self.generator_block(gen_model,style_model,256,3) 
      gen_model = self.generator_block(gen_model,style_model,128,2) 
      gen_model = self.generator_block(gen_model,style_model,64,2)
      gen_model = self.generator_block(gen_model,style_model,32,2)
      gen_model = self.generator_block(gen_model,style_model,16,2)
      return Conv2D(filters=self.channels, kernel_size=1, padding='same',activation='sigmoid')(gen_model)

  def build_discriminator(self,disc_model_input):                      
    disc_model = self.disc_conv_block(disc_model_input,4,2) 
    disc_model = self.disc_conv_block(disc_model,8,2)       
    disc_model = self.disc_conv_block(disc_model,16,4)        
    disc_model = self.disc_conv_block(disc_model,32,4)     
    disc_model = self.disc_conv_block(disc_model,64,8)     
    disc_model = self.disc_conv_block(disc_model,128,8)    
    disc_model = Flatten()(disc_model)
    disc_model = self.disc_dense_block(disc_model,1024)
    disc_model = self.disc_dense_block(disc_model,256,minibatch=True)
    
    disc_model = Dense(1,activation="sigmoid")(disc_model)
    return disc_model