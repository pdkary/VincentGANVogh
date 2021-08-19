from GanConfig import ActivationConfig
from typing import List, Tuple
from keras.layers import Layer,Input,Conv2D
import tensorflow as tf
from tensorflow.python.ops.gen_nn_ops import Conv2D

class MultiConv2D(Layer):
    def __init__(self,input_shape:Tuple,filters: List[int],convolutions:int,kernel_size:int,activation:ActivationConfig):
        super(MultiConv2D, self).__init__()
        self.W = self.add_weight(
            shape=(1,len(filters)),
            dtype=tf.float32,
            initializer='uniform',
            trainable=True)
        self.input = Input(shape=input_shape)
        self.kernel_size = kernel_size
        self.convolutions = convolutions
        self.activation = activation
        self.convs = [self.conv_block(f) for f in filters]

    def conv_block(self,filters):
        out = self.input
        for i in range(self.convolutions):
            out = Conv2D(filters,self.kernel_size,padding="same",kernel_initializer="he_normal")(out)
            out = self.activation.get()(out)
        return out

    ##inputs will just be an image tensor
    def call(self, inputs):
        weights = tf.nn.softmax(self.W, axis=-1)
        convolveds = [c(inputs) for c in self.convs]
        return weights*convolveds