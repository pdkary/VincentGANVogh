from typing import List
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from layers.CallableConfig import ActivationConfig, NoneCallable
from third_party_layers.MinibatchDiscrimination import MinibatchDiscrimination

class DenseModel():
    def __init__(self,
                 input: KerasTensor,
                 dense_layers: List[int],
                 activation: ActivationConfig = NoneCallable,
                 minibatch_size: int = 0,
                 dropout_rate: float = 0.0):
        self.input_layer = input
        self.dense_layers = dense_layers
        self.activation = activation
        self.minibatch_size = minibatch_size
        self.dropout_rate = dropout_rate

    def build(self):
        out = self.input_layer
        for i,l_size in enumerate(self.dense_layers):
            print("BLOCK SHAPE: ",out.shape)
            out = self.dense_block(out,i,l_size)
        print("BLOCK SHAPE: ",out.shape)
        return out

    def dense_block(self,input_tensor: KerasTensor,i:int,size:int):
        name = str(size) + "_" + str(i)
        out = input_tensor
        if self.minibatch_size > 0:
            out = MinibatchDiscrimination(self.minibatch_size,4)(out)
        out = Dense(size)(out)
        if self.dropout_rate > 0:
            out = Dropout(self.dropout_rate,name="dense_dropout_"+name)(out)
        out = self.activation.get()(out)
        return out
        
