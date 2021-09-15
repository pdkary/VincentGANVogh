from tensorflow.keras.layers import Layer
import keras.backend as K
import tensorflow as tf

class AdaptiveInstanceNormalization(Layer):
    def __init__(self):
        super(AdaptiveInstanceNormalization, self).__init__()

    def calc_mean_std(self,features):
        batch_size, h, w, c = features.shape
        reshaped_features = tf.reshape(features,(-1,h*w*c))
        features_mean = K.mean(reshaped_features,axis=1)
        features_mean = tf.reshape(features_mean,(-1, 1, 1, c))
        features_std = K.mean(reshaped_features,axis=1)
        features_std = tf.reshape(reshaped_features,(-1, 1, 1, c))
        return features_mean,features_std

    def call(self, content_features,style_features):
        content_mean,content_std = self.calc_mean_std(content_features)
        style_mean, style_std = self.calc_mean_std(style_features)
        normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
        return normalized_features