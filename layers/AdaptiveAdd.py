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
        print(conv_tensor.shape,noise_tensor.shape)
        weights = tf.nn.softmax(self.W, axis=-1)
        return tf.reduce_sum(conv_tensor + weights*noise_tensor, axis=-1) # (n_batch, n_feat) 