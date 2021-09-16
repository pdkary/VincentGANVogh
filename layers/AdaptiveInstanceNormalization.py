from tensorflow.keras.layers import Layer
import keras.backend as K
import tensorflow as tf

class AdaptiveInstanceNormalization(Layer):
    def __init__(self):
        super(AdaptiveInstanceNormalization, self).__init__()

    # def call(self, inputs):
    #     input_tensor, gamma, beta = inputs
    #     ## we want to get the mean of the channels, per batch
    #     ## [batch,height,width,channels]
    #     mean = K.mean(input_tensor, axis = [1, -1], keepdims = True)
    #     std = K.std(input_tensor, axis = [1, -1], keepdims = True) + 1e-7
    #     y = (input_tensor - mean) / std
        
    #     pool_shape = [-1, 1, 1, y.shape[-1]]
    #     scale = K.reshape(gamma, pool_shape)
    #     bias = K.reshape(beta, pool_shape)
    #     return y * scale + bias
    
    def calc_mean_std(self,features):
        mean = K.mean(features,axis=[1,4],keepdims=True)
        std = K.std(features,axis=[1,4],keepdims=True) + 1e-6
        return mean,std
        
    def call(self,content_features,style_features):
        ##style size: [4,256,256,3]
        ##content size: (could be many things)
        ##  - [4, 8  , 8,   512]
        ##  - [4, 16 , 16,  256]
        ##  - [4, 32 , 32,  128]
        ##  - [4, 64 , 64,  64 ]
        ##  - [4, 128, 128, 32 ]
        ##  - [4, 256, 256, 16 ]
        ##  - [4, 256, 256, 4  ]
        
        #style_mean size: []
        style_mean,style_std = self.calc_mean_std(style_features)
        print(style_mean.shape)
        
        content_mean,content_std = self.calc_mean_std(content_features)
        normalized_content = (content_features - content_mean)/content_std
        return style_std*normalized_content + style_mean
        