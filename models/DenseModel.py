from typing import List, Tuple, Union
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Reshape
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from layers.CallableConfig import ActivationConfig, NoneCallable
from inputs.GanInput import GanInput, LatentSpaceInput
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

class LatentSpaceModel(DenseModel):
    def __init__(self,
                 inner_layers: int,
                 inner_layer_size: int,
                 output_shape: Tuple[int,int,int],
                 activation: ActivationConfig):
        super().__init__(LatentSpaceInput([inner_layer_size]), [inner_layer_size for i in range(inner_layers)], activation)
        self.output_shape = output_shape
    
    def build(self):
        out = super().build()
        out = Dense(np.prod(self.output_shape))(out)
        out = self.activation.get()(out)
        out = Reshape(self.output_shape)(out)
        return out

    def get_training_batch(self,batch_size):
        return self.input.get_training_batch(batch_size)
    
    def get_validation_batch(self,batch_size):
        return self.input.get_validation_batch(batch_size)
