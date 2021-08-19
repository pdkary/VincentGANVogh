from keras.layers import Layer
import tensorflow as tf

class AdaptiveAdd(Layer):
    def __init__(self):
        super(AdaptiveAdd, self).__init__()
        self.W = self.add_weight(
            shape=(1,1),
            dtype=tf.float32,
            initializer='uniform',
            trainable=True)

    def call(self, inputs):
        conv_tensor,noise_tensor = inputs
        weights = tf.nn.softmax(self.W, axis=-1)
        out = conv_tensor + weights*noise_tensor
        return out