from tensorflow.keras.layers import Layer
import keras.backend as K
import tensorflow as tf

class AdaptiveInstanceNormalization(Layer):
    def __init__(self,**kwargs):
        super(AdaptiveInstanceNormalization, self).__init__(**kwargs)
    
    def call(self, inputs):
        input_tensor, beta, gamma = inputs
        mean = K.mean(input_tensor, axis = [0,1,2], keepdims = True)
        std = K.std(input_tensor, axis = [0,1,2], keepdims = True) + 1e-7
        normed = (input_tensor - mean) / std
        return normed * gamma + beta
        