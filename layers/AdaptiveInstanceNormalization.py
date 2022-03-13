import tensorflow as tf
from tensorflow.keras.layers import Layer
## aight ive seen a lot of shit like this before and never liked any of it so here's mine
def get_mean_std(x, epsilon=1e-5):
    axes = [1, 2]

    # Compute the mean and standard deviation of a tensor.
    mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    standard_deviation = tf.sqrt(variance + epsilon)
    return mean, standard_deviation

class AdaptiveInstanceNormalization(Layer):
    def __init__(self, size=256):
        super(AdaptiveInstanceNormalization,self).__init__()
        w_init = tf.random_normal_initializer()
        self.M = tf.Variable(
            initial_value=w_init(shape=(1,size,size), dtype="float32"),trainable=True
        )
    
    def call(self,inputs):
        content_mean, content_std = get_mean_std(inputs)
        style_mean, style_std = get_mean_std(self.M)
        t = style_std * (inputs - content_mean) / content_std + style_mean
        return t
