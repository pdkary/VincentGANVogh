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
    

def adain(content_features,style_features,axis=[1,2]):
    content_mean = K.mean(content_features,axis,keepdims=True)
    content_std = K.std(content_features,axis,keepdims=True) + 1e-7
    style_mean = K.mean(style_features,axis,keepdims=True)
    style_std = K.std(style_features,axis,keepdims=True) + 1e-7
    normed = (content_features - content_mean)/content_std
    out = style_std*normed + style_mean
    return out
        