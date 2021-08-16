from GanConfig import DiscriminatorModelConfig
from MinibatchDiscrimination import MinibatchDiscrimination
from keras.layers import Dense,Dropout,LeakyReLU,Conv2D,MaxPooling2D,Flatten, Input
from keras.models import Model

class Discriminator(DiscriminatorModelConfig):
    def __init__(self,disc_model_config):
        DiscriminatorModelConfig.__init__(self,**disc_model_config.__dict__)
        self.input = Input(shape=self.img_shape, name="image_input")
        
    def build(self):
        out = self.input
        for shape in self.disc_layer_shapes:
            out = self.disc_conv_block(out,*shape) 
        
        out = Flatten()(out)
        if self.minibatch:
            out = MinibatchDiscrimination(self.minibatch_size,self.img_shape[-1])(out)
        
        for size,dropout in zip(self.disc_dense_sizes,self.disc_layer_dropout):
            out = self.disc_dense_block(out,size,dropout=dropout)
        out = Dense(1,activation="sigmoid", kernel_initializer = 'he_normal')(out)
        
        disc_model = Model(inputs=self.input,outputs=out,name="Discriminator")
        disc_model.compile(optimizer=self.disc_optimizer,
                           loss=self.disc_loss_function,
                           metrics=['accuracy'])
        disc_model.summary()
        return disc_model
    
    def disc_dense_block(self,input_tensor,size,dropout=True):
        out_db = Dense(size, kernel_initializer = 'he_normal')(input_tensor)
        out_db = Dropout(self.dropout_rate)(out_db) if dropout else out_db
        out_db = self.convolution_activation(out_db)
        return out_db
    
    def disc_conv_block(self,input_tensor, filters, convolutions):
        out_cb = input_tensor
        for i in range(convolutions):
            out_cb = Conv2D(filters,self.disc_kernel_size,padding="same")(out_cb)
            out_cb = self.normalization_layer(out_cb)
            out_cb = self.convolution_activation(out_cb)
        out_cb = MaxPooling2D()(out_cb)
        return out_cb
    