from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf

class AdaptiveInstanceNormalization(Layer):
    def __init__(self,size,name,**kwargs):
        super(AdaptiveInstanceNormalization, self).__init__(**kwargs)
        self.size=size
        self.layer_name=name
    
    def build(self,input_shape):
        self.B = self.add_weight(
            shape=input_shape[-1],
            dtype=tf.float32,
            initializer='ones',
            trainable=True,
            name=self.layer_name + "_beta"
        )
        self.G = self.add_weight(
            shape=input_shape[-1],
            dtype=tf.float32,
            initializer='zeros',
            trainable=True,
            name=self.layer_name + "_gamma"
        )
        
    def call(self, input_tensor):
        i_shape = input_tensor.shape[-1]
        beta = K.reshape(self.B,(-1,1,1,i_shape))
        gamma = K.reshape(self.G,(-1,1,1,i_shape))
        mean = K.mean(input_tensor, axis = [1,2], keepdims = True)
        std = K.std(input_tensor, axis = [1,2], keepdims = True) + 1e-7
        normed = (input_tensor - mean)/std
        return [normed * gamma + beta, self.B,self.G]