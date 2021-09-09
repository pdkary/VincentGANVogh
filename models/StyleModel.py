from tensorflow.keras.regularizers import L2
from config.GeneratorConfig import StyleModelConfig
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
import tensorflow as tf

class StyleModel(StyleModelConfig):
    def __init__(self,style_config):
        super().__init__(**style_config.__dict__)
        self.input = Input(shape=self.style_latent_size, name="style_model_input")
        self.model = self.input
        for i in range(self.style_layers):
            self.model = Dense(self.style_latent_size, kernel_regularizer=L2(), kernel_initializer = 'he_normal')(self.model)
            self.model = self.style_activation.get()(self.model)
    
    def get_batch(self,batch_size:int):
        return tf.random.normal(shape = (batch_size,self.style_latent_size),dtype=tf.float32)
        