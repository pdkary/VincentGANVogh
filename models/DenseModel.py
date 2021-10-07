from typing import List, Union
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.python.keras.models import Functional
from layers.CallableConfig import ActivationConfig, NoneCallable
from inputs.GanInput import GanInput
from third_party_layers.MinibatchDiscrimination import MinibatchDiscrimination

class DenseModel():
    def __init__(self,
                 dense_input: Union[Functional,GanInput],
                 dense_layers: List[int],
                 activation: ActivationConfig = NoneCallable,
                 minibatch_size: int = 0,
                 dropout_rate: float = 0.0):
        self.dense_input = dense_input.input if isinstance(dense_input,GanInput) else dense_input
        self.inputs = [self.dense_input]
        self.dense_layers = dense_layers
        self.activation = activation
        self.minibatch_size = minibatch_size
        self.dropout_rate = dropout_rate

    def build(self):
        model = self.dense_input
        for i,l_size in enumerate(self.dense_layers):
            name = str(l_size) + "_" + str(i)
            if self.minibatch_size > 0:
                model = MinibatchDiscrimination(self.minibatch_size,4)(model)
            model = Dense(l_size)(model)
            model = Dropout(self.dropout_rate,name="dense_dropout_" + name)(model)
            model = self.activation.get()(model)
        return model