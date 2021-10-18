from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf

class AdaptiveInstanceNormalization(Layer):
    def __init__(self,size,name,**kwargs):
        super(AdaptiveInstanceNormalization, self).__init__(**kwargs)
        self.size=size
        self.layer_name=name
    
    def build(self,input_shape):
        c_shape,s_shape = input_shape
        mat_shape = (c_shape[-1],s_shape[-1])

        self.B = self.add_weight(
            shape=mat_shape,
            dtype=tf.float32,
            initializer='ones',
            trainable=True,
            name=self.layer_name + "_beta"
        )
        self.G = self.add_weight(
            shape=mat_shape,
            dtype=tf.float32,
            initializer='zeros',
            trainable=True,
            name=self.layer_name + "_gamma"
        )
        
    def call(self, input_tensor):
        c_input,s_input = input_tensor
        beta = tf.transpose(tf.linalg.matmul(self.B,s_input,transpose_b=True))
        beta = K.reshape(beta,(-1,1,1,beta.shape[-1]))
        gamma = tf.transpose(tf.linalg.matmul(self.G,s_input,transpose_b=True))
        gamma = K.reshape(gamma,(-1,1,1,gamma.shape[-1]))
        mean = K.mean(c_input, axis = [1,2], keepdims = True)
        std = K.std(c_input, axis = [1,2], keepdims = True) + 1e-7
        normed = (c_input - mean)/std
        out = normed*gamma + beta
        return [out, beta, gamma]
