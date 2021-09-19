from tensorflow.keras.layers import Layer
import keras.backend as K

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
    

def adain(content_features,style_features):
    print("content_features: ",content_features.shape)
    print("style_features: ",style_features.shape)
    content_mean = K.mean(content_features,axis=[1,2],keepdims=True)
    print("content_mean: ",content_mean.shape)
    content_std = K.std(content_features,axis=[1,2],keepdims=True) + 1e-7
    print("content_std: ",content_std.shape)
    style_mean = K.mean(style_features,axis=[1,2],keepdims=True)
    print("style_mean: ",style_mean.shape)
    style_std = K.std(style_features,axis=[1,2],keepdims=True) + 1e-7
    print("style_std: ",style_std.shape)
    normed = (content_features - content_mean)/content_std
    print("normed: ",normed.shape)
    out = style_std*normed + style_mean
    print("out: ",out.shape)
    return out
        