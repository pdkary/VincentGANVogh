from GanConfig import DiscConvLayerConfig, DiscDenseLayerConfig, DiscriminatorModelConfig
from third_party_layers.MinibatchDiscrimination import MinibatchDiscrimination
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,Input
from keras.models import Functional, Model

class Discriminator(DiscriminatorModelConfig):
    def __init__(self,disc_model_config: DiscriminatorModelConfig):
        DiscriminatorModelConfig.__init__(self,**disc_model_config.__dict__)
        self.input = Input(shape=self.img_shape, name="image_input")
        
    def build(self):
        out = self.input
        for layer_config in self.disc_conv_layers:
            out = self.disc_conv_block(out,layer_config) 
        
        out = Flatten()(out)
        if self.minibatch:
            out = MinibatchDiscrimination(self.minibatch_size,self.img_shape[-1])(out)
        
        for layer_config in self.disc_dense_layers:
            out = self.disc_dense_block(out,layer_config)
        out = Dense(1,activation="sigmoid", kernel_initializer = 'he_normal')(out)
        
        disc_model = Model(inputs=self.input,outputs=out,name="Discriminator")
        disc_model.compile(optimizer=self.disc_optimizer,
                           loss=self.disc_loss_function,
                           metrics=['accuracy'])
        disc_model.summary()
        return disc_model
    
    def disc_dense_block(self,input_tensor: Functional,config: DiscDenseLayerConfig):
        out_db = Dense(config.size, kernel_initializer = 'he_normal')(input_tensor)
        out_db = Dropout(config.dropout_rate)(out_db) if config.dropout_rate > 0 else out_db
        out_db = config.activation.get()(out_db)
        return out_db
    
    def disc_conv_block(self,input_tensor: Functional,config: DiscConvLayerConfig):
        out_cb = input_tensor
        for i in range(config.convolutions):
            out_cb = Conv2D(config.filters,config.kernel_size,padding="same")(out_cb)
            out_cb = config.normalization(out_cb)
            out_cb = config.activation.get()(out_cb)
        out_cb = MaxPooling2D()(out_cb)
        return out_cb
    