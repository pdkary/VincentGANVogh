from tensorflow.keras.layers import Layer
import tensorflow as tf
import tensorflow.keras.backend as K

class AdaptiveAdd(Layer):
    def __init__(self):
        super(AdaptiveAdd, self).__init__()
        self.W = self.add_weight(
            shape=(1,1),
            dtype=tf.float32,
            initializer='zeros',
            trainable=True,
            name="add_coefficient")

    def call(self, inputs):
        conv_tensor,noise_tensor = inputs
        noise_tensor = (noise_tensor - K.mean(noise_tensor))/K.std(noise_tensor)
        weights = tf.nn.sigmoid(self.W, axis=-1)
        out = conv_tensor + weights*noise_tensor
        return out