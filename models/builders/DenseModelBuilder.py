from typing import Tuple
from tensorflow.keras.layers import Dense, Dropout, Reshape
from config.CallableConfig import ActivationConfig
from models.builders.BuilderBase import BuilderBase
from third_party_layers.MinibatchDiscrimination import MinibatchDiscrimination
import numpy as np

class DenseModelBuilder(BuilderBase):

    def reshape(self,shape: Tuple[int,int,int]):
        self.out = Dense(np.prod(shape))(self.out)
        self.out = Reshape(shape)(self.out)
        return self

    def layer(self,size):
        self.out = Dense(size)(self.out)
        return self.out
    
    def track(self):
        pass

    def block(self,size: int, activation: ActivationConfig, 
                    dropout_rate=0.0, minibatch_size=0, minibatch_dim=4):
        name = str(size) + "_" + str(self.layer_count)
        self.out = MinibatchDiscrimination(minibatch_size,minibatch_dim)(self.out) if minibatch_size > 0 else self.out
        self.out = self.layer(size)
        self.out = Dropout(dropout_rate,name="dense_dropout_"+name)(self.out) if dropout_rate > 0.0 else self.out
        self.out = activation.get()(self.out)
        self.layer_count += 1
        return self
