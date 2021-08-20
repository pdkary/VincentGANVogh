from keras.layers import Layer, Activation,Cropping2D,Conv2D
from keras.models import Functional
import tensorflow as tf
from layers.AdaptiveAdd import AdaptiveAdd
'''
inputs: a,b
    - crop b to a 2d size
    - convolve b to match channels
    - adaptive add a + (z)b
        - where z is trained
'''
class CropConvAdd(Layer):
    def __init__(self,filters:int,kernel_size:int):
        super(CropConvAdd, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size        

    def call(self, inputs):
        a,b = inputs
        a_size,b_size = a.shape[1],b.shape[1]
        assert (b_size > a_size),"input[1] must be larger than input[2]"
        b = Cropping2D((b_size-a_size)//2)(b)
        b = Conv2D(self.filters,self.kernel_size,padding='same',kernel_initializer='he_normal')(b)
        return AdaptiveAdd()([a,b])