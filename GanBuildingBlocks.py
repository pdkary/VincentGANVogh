import keras.backend as K
from keras.layers import Dense,UpSampling2D,Conv2D,Lambda,BatchNormalization,Add,Dropout,MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from MinibatchDiscrimination import MinibatchDiscrimination

def AdaIN(input_arr):
  input_tensor, gamma, beta = input_arr
  mean = K.mean(input_tensor, axis = [1, 2], keepdims = True)
  std = K.std(input_tensor, axis = [1, 2], keepdims = True) + 1e-7
  y = (input_tensor - mean) / std
  
  pool_shape = [-1, 1, 1, y.shape[-1]]
  scale = K.reshape(gamma, pool_shape)
  bias = K.reshape(beta, pool_shape)
  return y * scale + bias
  
class GanBuildingBlocks():
  def __init__(self,img_shape,kernel_size,relu_alpha,dropout_rate,batch_norm_momentum,noise_dict):
    self.img_shape = img_shape
    self.img_size = self.img_shape[1]
    self.channels = self.img_shape[-1]
    self.kernel_size = kernel_size
    self.relu_alpha = relu_alpha
    self.dropout_rate = dropout_rate
    self.batch_norm_momentum = batch_norm_momentum
    self.noise_dict = noise_dict
  
  def style_model_block(self,input_tensor,size):
    out = Dense(size, kernel_initializer = 'he_normal')(input_tensor)
    out = LeakyReLU(self.relu_alpha)(out)
    return out 

  def generator_block(self,input_tensor,style_model,filters,convolutions,upsampling=True,style=True,noise=True):
    out = input_tensor
    out = UpSampling2D(interpolation='bilinear')(out) if upsampling else out
    noise_model = self.noise_dict[out.shape[1]]
    for i in range(convolutions):
      if noise:
        noise_model = Conv2D(filters,1,padding='same',kernel_initializer='he_normal')(noise_model)
      if style:
        gamma = Dense(filters,bias_initializer='ones')(style_model)
        beta = Dense(filters,bias_initializer='zeros')(style_model)
      
      out = Conv2D(filters,self.kernel_size,padding='same', kernel_initializer = 'he_normal')(out)
      out = Lambda(AdaIN)([out,gamma,beta]) if style else BatchNormalization(momentum=self.batch_norm_momentum)(out)
      out = Add()([out,noise_model]) if noise else out
      out = LeakyReLU(self.relu_alpha)(out)
    return out

  def disc_dense_block(self,input_tensor,size,dropout=True,minibatch=False):
    out_db = Dense(size, kernel_initializer = 'he_normal')(input_tensor)
    out_db = Dropout(self.dropout_rate)(out_db) if dropout else out_db
    out_db = LeakyReLU(self.relu_alpha)(out_db)
    out_db = MinibatchDiscrimination(size, self.img_size)(out_db) if minibatch else out_db
    return out_db

  def disc_conv_block(self,input_tensor, filters, convolutions):
    out_cb = input_tensor
    for i in range(convolutions):
      out_cb = Conv2D(filters,self.kernel_size,padding="same")(out_cb)
      out_cb = BatchNormalization(momentum=self.batch_norm_momentum)(out_cb)
      out_cb = LeakyReLU(self.relu_alpha)(out_cb)
    out_cb = MaxPooling2D()(out_cb)
    return out_cb