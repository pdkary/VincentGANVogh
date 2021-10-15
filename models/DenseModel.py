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
        self.model = self.input_layer
        for i,l_size in enumerate(self.dense_layers):
            self.dense_block(i,l_size)
        return self.model

    def dense_block(self,i:int,size:int):
        name = str(size) + "_" + str(i)
        if self.minibatch_size > 0:
            self.model = MinibatchDiscrimination(self.minibatch_size,4)(self.model)
        self.model = Dense(size)(self.model)
        if self.dropout_rate > 0:
            self.model = Dropout(self.dropout_rate,name="dense_dropout_"+name)(self.model)
        self.model = self.activation.get()(self.model)
        
