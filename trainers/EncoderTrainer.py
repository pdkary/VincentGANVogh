from layers.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization
from tensorflow.python.keras.layers import Dense
from config.GeneratorConfig import StyleModelConfig
from models.StyleModel import StyleModel
from models.Generator import Generator
from models.Discriminator import Discriminator
from config.CallableConfig import ActivationConfig, NormalizationConfig
from layers.GanInput import RealImageInput
from config.TrainingConfig import DataConfig
from config.TrainingConfig import GanTrainingConfig
from typing import List
from models.Encoder import get_decoder,get_encoder
from tensorflow.python.keras.layers import BatchNormalization,LeakyReLU,Activation
from tensorflow.keras.optimizers import Adam
from third_party_layers.InstanceNormalization import InstanceNormalization
import numpy as np
import tensorflow as tf

##activations
leakyRELU_dense = ActivationConfig(LeakyReLU,"dense_relu",dict(alpha=0.1))
leakyRELU_conv = ActivationConfig(LeakyReLU,"conv_relu",dict(alpha=0.08))
sigmoid = ActivationConfig(Activation,dict(activation="sigmoid"))
linear = ActivationConfig(Activation,dict(activation="linear"))
##normalizations
instance_norm = NormalizationConfig(InstanceNormalization)
batch_norm = NormalizationConfig(BatchNormalization,dict(momentum=0.5))
##optimizers
img_shape = (256,256,3)

class EncoderTrainer(GanTrainingConfig):
    def __init__(self,
               gan_training_config: GanTrainingConfig,
               content_data_source: DataConfig,
               style_data_source: DataConfig):
        GanTrainingConfig.__init__(self,**gan_training_config.__dict__)
        assert self.gen_batch_size == self.disc_batch_size, "Batch sizes must be equal"
        self.preview_size = self.preview_cols*self.preview_rows
        self.content_data_source = RealImageInput(content_data_source)
        self.style_data_source = RealImageInput(style_data_source)
        self.encoder = Discriminator(get_encoder(3,1000,leakyRELU_conv,leakyRELU_dense,sigmoid,instance_norm,Adam(learning_rate=0.002)))
        self.encoder_model = self.encoder.build()
        self.decoder = Generator(get_decoder(3,1000,leakyRELU_conv,sigmoid,batch_norm,Adam(learning_rate=0.02)))
        self.decoder_model = self.decoder.build()

    def train_step(self):
        content_input = self.content_data_source.get_batch(self.gen_batch_size)
        style_input = self.style_data_source.get_batch(self.gen_batch_size)
        with tf.GradientTape() as decoder_tape, tf.GradientTape() as encoder_tape:
            
            encoded_content = self.encoder_model(content_input)
            encoded_style = self.encoder_model(style_input)
            
            gamma = Dense(1000,bias_initializer='ones')(encoded_style)
            beta = Dense(1000,bias_initializer='zeros')(encoded_style)
            adapted_encoding = AdaptiveInstanceNormalization()([encoded_content,gamma,beta])
            output = self.decoder_model(adapted_encoding)
            encoded_output = self.encoder_model(output)
            
            eloss,eavg = self.encoder.loss_function(encoded_style,encoded_output)
            dloss,davg = self.decoder.loss_function(encoded_content,encoded_output)
            
            gradients_of_decoder = decoder_tape.gradient(dloss,self.decoder.trainable_variables)
            self.decoder.gen_optimizer.apply_gradients(zip(gradients_of_decoder,self.decoder.trainable_variables))
            
            gradients_of_encoder = encoder_tape.gradient(eloss,self.encoder.trainable_variables)
            self.encoder.disc_optimizer.apply_gradients(zip(gradients_of_encoder,self.encoder.trainable_variables))
        return eloss,eavg,dloss,davg
    
    def train(self,epochs,batches_per_epoch,printerval):
        for epoch in range(epochs):
            if self.plot:
                self.gan_plotter.start_epoch()
        
            for i in range(batches_per_epoch):
                eloss,eavg,dloss,davg = self.train_step()
                if self.plot:
                    self.gan_plotter.batch_update([eloss,eavg,dloss,davg])
        
            if epoch % printerval == 0:
                preview_content = self.content_data_source.get_batch(self.preview_size)
                preview_style = self.style_data_source.get_batch(self.preview_size)

                encoded_content = self.encoder_model(preview_content)
                encoded_style = self.encoder_model(preview_style)
                gamma = Dense(1000,bias_initializer='ones')(encoded_style)
                beta = Dense(1000,bias_initializer='zeros')(encoded_style)
                adapted_encoding = AdaptiveInstanceNormalization()([encoded_content,gamma,beta])
                
                generated_images = np.array(self.decoder_model.predict(adapted_encoding))
                self.content_data_source.save(epoch,generated_images,self.preview_rows,self.preview_cols,self.preview_margin)

            if epoch >= 10 and self.plot:
                self.gan_plotter.log_epoch()

    def train_n_eras(self,eras,epochs,batches_per_epoch,printerval,ma_size):
        if self.plot:
            from helpers.GanPlotter import GanPlotter
            self.gan_plotter = GanPlotter(moving_average_size=ma_size,labels=["ELoss","EAvg","DLoss","Davg","Epoch time")
        for i in range(eras):
            self.train(epochs,batches_per_epoch,printerval)
            filename = self.image_sources[0].data_helper.model_name + "%d"%((i+1)*epochs)
            print(self.model_output_path + filename)
            self.generator.save(self.model_output_path + filename)
