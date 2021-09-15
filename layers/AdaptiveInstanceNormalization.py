from tensorflow.keras.layers import Layer
import keras.backend as K

class AdaptiveInstanceNormalization(Layer):
    def __init__(self):
        super(AdaptiveInstanceNormalization, self).__init__()

    def calc_mean_std(self,features):
        batch_size, c = features.size()[:2]
        reshaped_features = K.reshape(features,(batch_size,c,-1))
        features_mean = K.mean(reshaped_features)
        features_mean = K.reshape(features_mean,(batch_size,c,1,1))
        features_std = K.std(reshaped_features)
        features_std = K.reshape(features_std,(batch_size,c,1,1)) + 1e-6
        return features_mean,features_std
        

    def call(self, content_features,style_features):
        content_mean,content_std = self.calc_mean_std(content_features)
        style_mean, style_std = self.calc_mean_std(style_features)
        normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
        return normalized_features