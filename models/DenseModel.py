from typing import List, Union
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from layers.CallableConfig import ActivationConfig, NoneCallable
from inputs.GanInput import GanInput
from third_party_layers.MinibatchDiscrimination import MinibatchDiscrimination

class DenseModel():
    def __init__(self,
                 input: Union[KerasTensor,GanInput],
                 dense_layers: List[int],
                 activation: ActivationConfig = NoneCallable,
                 minibatch_size: int = 0,
                 dropout_rate: float = 0.0):
        self.input = input
        self.inputs = [input.input] if isinstance(input,GanInput) else [input]
        self.dense_layers = dense_layers
        self.activation = activation
        self.minibatch_size = minibatch_size
        self.dropout_rate = dropout_rate

    def build(self):
        model = self.inputs[0]
        for i,l_size in enumerate(self.dense_layers):
            name = str(l_size) + "_" + str(i)
            if self.minibatch_size > 0:
                model = MinibatchDiscrimination(self.minibatch_size,4)(model)
            model = Dense(l_size)(model)
            model = Dropout(self.dropout_rate,name="dense_dropout_" + name)(model)
            model = self.activation.get()(model)
        return model