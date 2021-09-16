from tensorflow.keras.layers import Layer
import keras.backend as K

class AdaptiveInstanceNormalization(Layer):
    def __init__(self):
        super(AdaptiveInstanceNormalization, self).__init__()

    def call(self, inputs):
        input_tensor, gamma, beta = inputs
        mean = K.mean(input_tensor, axis = [1, -1], keepdims = True)
        std = K.std(input_tensor, axis = [1, -1], keepdims = True) + 1e-7
        y = (input_tensor - mean) / std
        
        pool_shape = [-1, 1, 1, y.shape[-1]]
        scale = K.reshape(gamma, pool_shape)
        bias = K.reshape(beta, pool_shape)
        return y * scale + bias