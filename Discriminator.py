from GanConfig import DiscriminatorModelConfig
from InstanceNormalization import InstanceNormalization
from MinibatchDiscrimination import MinibatchDiscrimination
from keras.layers import Dense,Dropout,LeakyReLU,Conv2D,MaxPooling2D,Flatten, Input

class Discriminator(DiscriminatorModelConfig):
    def __init__(self,disc_model_config):
        self.input = Input(shape=self.img_shape, name="image_input")
        DiscriminatorModelConfig.__init__(self,**disc_model_config.__data__)
        
    def build(self):
        disc_model = self.input
        for shape in self.disc_layer_shapes:
            disc_model = self.disc_conv_block(disc_model,*shape) 
        
        disc_model = Flatten()(disc_model)
        if self.minibatch:
            disc_model = MinibatchDiscrimination(self.minibatch_size,self.img_shape[-1])(disc_model)
        
        for size,dropout in zip(self.disc_dense_sizes,self.disc_layer_dropout):
            disc_model = self.disc_dense_block(disc_model,size,dropout=dropout)
            disc_model = Dense(1,activation="sigmoid", kernel_initializer = 'he_normal')(disc_model)
        return disc_model
    
    def disc_dense_block(self,input_tensor,size,dropout=True):
        out_db = Dense(size, kernel_initializer = 'he_normal')(input_tensor)
        out_db = Dropout(self.dropout_rate)(out_db) if dropout else out_db
        out_db = LeakyReLU(self.relu_alpha)(out_db)
        return out_db
    
    def disc_conv_block(self,input_tensor, filters, convolutions):
        out_cb = input_tensor
        for i in range(convolutions):
            out_cb = Conv2D(filters,self.disc_kernel_size,padding="same")(out_cb)
            out_cb = InstanceNormalization()(out_cb)
            out_cb = LeakyReLU(self.relu_alpha)(out_cb)
        out_cb = MaxPooling2D()(out_cb)
        return out_cb
    