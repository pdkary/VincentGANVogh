from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from layers.CallableConfig import ActivationConfig
from third_party_layers.MinibatchDiscrimination import MinibatchDiscrimination

class DenseModelBuilder():
    def __init__(self,input_layer: KerasTensor):
        self.layer_count = 0
        self.out = input_layer

    def build(self):
        return self.out

    def dense_layer(self,size: int, activation: ActivationConfig, 
                    dropout_rate=0.0, minibatch_size=0, minibatch_dim=4):
        name = str(size) + "_" + str(self.layer_count)
        self.out = MinibatchDiscrimination(minibatch_size,minibatch_dim)(self.out) if minibatch_size > 0 else self.out
        self.out = Dense(size)(self.out)
        self.out = Dropout(dropout_rate,name="dense_dropout_"+name)(self.out) if dropout_rate > 0.0 else self.out
        self.out = activation.get()(self.out)
        self.layer_count += 1
        return self
    
        
