from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
## aight ive seen a lot of shit like this before and never liked any of it so here's mine

class AdaptiveInstanceNormalization(Layer):
    def __init__(self,axis=None,epsilon=1e-5):
        self.axis = axis
        self.epsilon=epsilon
        super(AdaptiveInstanceNormalization,self).__init__()
    
    def build(self,input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')
        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        shape = (1,) if self.axis is None else (input_shape[self.axis],)
        
        self.gamma = self.add_weight(
            name="gamma",
            shape=shape,
            initializer="random_normal",
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=shape,
            initializer="random_normal",
            trainable=True)
    
    def call(self,inputs):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))
        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]
        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean)/stddev
        broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
        broadcast_beta = K.reshape(self.beta, broadcast_shape)
        return normed * broadcast_gamma + broadcast_beta

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
        }
        base_config = super(AdaptiveInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
