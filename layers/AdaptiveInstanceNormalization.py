from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class AdaptiveInstanceNormalization(Layer):
    def __init__(self,**kwargs):
        super(AdaptiveInstanceNormalization, self).__init__(**kwargs)
    
    def call(self, inputs):
        input_tensor, beta, gamma = inputs
        beta = K.reshape(beta,(-1,1,1,beta.shape[-1]))
        gamma = K.reshape(gamma,(-1,1,1,gamma.shape[-1]))
        mean = K.mean(input_tensor, axis = [1,2], keepdims = True)
        std = K.std(input_tensor, axis = [1,2], keepdims = True) + 1e-7
        normed = (input_tensor - mean)/std
        return normed * gamma + beta
        