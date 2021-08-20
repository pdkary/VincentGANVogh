from keras.layers import Layer, Dense, Lambda
import tensorflow as tf
import keras.backend as K

class AdaptiveInstanceNormalization(Layer):
    def __init__(self,filters):
        super(AdaptiveInstanceNormalization, self).__init__()
        self.filters = filters

    def call(self, inputs):
        input_model,style_model = inputs
        gamma = Dense(self.filters,bias_initializer='ones')(style_model)
        beta = Dense(self.filters,bias_initializer='zeros')(style_model)
        return Lambda(self.AdaIN)([input_model,gamma,beta])

    def AdaIN(self,input_arr):
        input_tensor, gamma, beta = input_arr
        mean = K.mean(input_tensor, axis = [1, 2], keepdims = True)
        std = K.std(input_tensor, axis = [1, 2], keepdims = True) + 1e-7
        y = (input_tensor - mean) / std
        
        pool_shape = [-1, 1, 1, y.shape[-1]]
        scale = K.reshape(gamma, pool_shape)
        bias = K.reshape(beta, pool_shape)
        return y * scale + bias